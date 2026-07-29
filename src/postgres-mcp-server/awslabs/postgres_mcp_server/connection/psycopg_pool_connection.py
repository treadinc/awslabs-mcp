# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Psycopg connector for postgres MCP Server.

This connector provides direct connection to PostgreSQL databases using psycopg.
It supports both Aurora PostgreSQL and RDS PostgreSQL instances via direct connection
parameters (host, port, database, user, password) or via AWS Secrets Manager.
"""

import asyncio
import boto3
import json
import re
from aiorwlock import RWLock
from awslabs.postgres_mcp_server import __user_agent__
from awslabs.postgres_mcp_server.connection.abstract_db_connection import AbstractDBConnection
from botocore.config import Config
from datetime import datetime, timedelta
from loguru import logger
from psycopg_pool import AsyncConnectionPool
from typing import Any, Dict, List, Optional, Tuple


def get_credentials_from_secret(
    secret_arn: str, region: str, is_test: bool = False
) -> Tuple[str, str]:
    """Fetch database (username, password) from AWS Secrets Manager.

    The secret payload is expected to be JSON with ``username``/``user``/
    ``Username`` and ``password``/``Password`` fields — the standard shape
    RDS-managed secrets produce.

    Module-level so callers (e.g. ``server.internal_create_connection``
    on the IAM-auth path, which only needs the username) can reuse it
    without instantiating a connection pool.

    Args:
        secret_arn: Secrets Manager ARN.
        region: AWS region of the secret.
        is_test: If True, return deterministic test credentials instead of
            hitting AWS. Intended for unit tests only.

    Returns:
        ``(username, password)`` tuple.

    Raises:
        ValueError: If the secret is missing, malformed, or lacks
            username/password fields.
    """
    if is_test:
        return 'test_user', 'test_password'

    try:
        logger.debug(f'Creating Secrets Manager client in region {region}')
        session = boto3.Session()
        client = session.client(service_name='secretsmanager', region_name=region)

        logger.debug(f'Retrieving secret value for {secret_arn}')
        get_secret_value_response = client.get_secret_value(SecretId=secret_arn)
        logger.debug('Successfully retrieved secret value')

        if 'SecretString' not in get_secret_value_response:
            logger.error('Secret does not contain a SecretString')
            raise ValueError('Secret does not contain a SecretString')

        secret = json.loads(get_secret_value_response['SecretString'])
        logger.debug(f'Secret keys: {", ".join(secret.keys())}')

        username = secret.get('username') or secret.get('user') or secret.get('Username')
        password = secret.get('password') or secret.get('Password')

        if not username:
            logger.error(
                f'Username not found in secret. Available keys: {", ".join(secret.keys())}'
            )
            raise ValueError(
                f'Secret does not contain username. Available keys: {", ".join(secret.keys())}'
            )

        if not password:
            logger.error('Password not found in secret')
            raise ValueError(
                f'Secret does not contain password. Available keys: {", ".join(secret.keys())}'
            )

        logger.debug(f'Successfully extracted credentials for user: {username}')
        return username, password
    except Exception as e:
        logger.exception(f'Failed to retrieve credentials from Secrets Manager: {str(e)}')
        raise ValueError(f'Failed to retrieve credentials from Secrets Manager: {str(e)}')


class PsycopgPoolConnection(AbstractDBConnection):
    """Class that wraps DB connection using psycopg connection pool.

    This class can connect directly to any PostgreSQL database, including:
    - Aurora PostgreSQL (using the cluster endpoint)
    - RDS PostgreSQL (using the instance endpoint)
    - Self-hosted PostgreSQL

    It uses AWS Secrets Manager (secret_arn and region) for authentication.
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        readonly: bool,
        secret_arn: str,
        db_user: str,
        region: str,
        is_iam_auth: bool = False,
        pool_expiry_min: int = 30,
        min_size: int = 0,
        max_size: int = 10,
        is_test: bool = False,
        connect_host: Optional[str] = None,
        connect_port: Optional[int] = None,
    ):
        """Initialize a new DB connection pool.

        Args:
            host: Database host (Aurora cluster endpoint or RDS instance endpoint). Used to
                sign the IAM auth token (get_iam_auth_token) and as the connection-map
                identity; must be the real endpoint even when the socket is opened
                elsewhere (see connect_host).
            port: Database port. Used for IAM token signing (the port RDS listens on); must
                be the real DB port (typically 5432) even when connect_port differs.
            database: Database name
            readonly: Whether connections should be read-only
            secret_arn: ARN of the secret containing credentials
            db_user: Database username
            region: AWS region for Secrets Manager
            is_iam_auth: Whether to use IAM authentication
            pool_expiry_min: Pool expiry time in minutes
            min_size: Minimum number of connections kept open at all times. Defaults to 0 so
                a session that isn't actively querying holds no server-side connection — each
                MCP client process gets its own pool for the life of the session, so a nonzero
                min_size here multiplies by however many sessions are open concurrently and
                stays claimed for hours, which is what exhausts a per-role CONNECTION LIMIT.
                With min_size=0, psycopg_pool's own idle reaper closes an unused connection
                after max_idle (default 10 min) instead of holding it indefinitely.
            max_size: Maximum number of connections in the pool
            is_test: Whether this is a test connection
            connect_host: Host to actually open the TCP socket to. Defaults to host. Set
                this (e.g. 127.0.0.1) when reaching the DB through an SSH tunnel/proxy
                while the IAM token stays signed for the real endpoint (host).
            connect_port: Port to actually open the TCP socket to. Defaults to port. Set
                this (e.g. a local tunnel port) alongside connect_host.
        """
        super().__init__(readonly)
        self.host = host
        self.port = port
        # Socket target. Defaults to the real endpoint; override to reach the DB through a
        # tunnel/proxy. The IAM token (get_iam_auth_token) is still signed for host/port.
        self.connect_host = connect_host or host
        self.connect_port = connect_port or port
        self.database = database
        self.min_size = min_size
        self.max_size = max_size
        self.region = region
        self.is_iam_auth = is_iam_auth
        self.user = db_user
        self.pool_expiry_min = pool_expiry_min
        self.secret_arn = secret_arn
        self.is_test = is_test
        self.pool: Optional['AsyncConnectionPool[Any]'] = None
        self.rw_lock = RWLock()
        self.created_time = datetime.now()

        if is_iam_auth:
            # if db_user is set, then it is IAM auth scenario and iam_auth_token must be set
            if not db_user:
                raise ValueError('db_user must be set when is_iam_auth is True')

            # set pool expiry before IAM auth token expiry of 15 minutes
            self.pool_expiry_min = 14
            logger.debug(f'Use IAM auth for user: {db_user}')

    async def initialize_pool(self):
        """Initialize the connection pool."""
        async with self.rw_lock.reader_lock:
            if self.pool is not None:
                return

        async with self.rw_lock.writer_lock:
            if self.pool is not None:
                return

            logger.debug(
                f'initialize_pool:\n'
                f'endpoint:{self.host}\n'
                f'port:{self.port}\n'
                f'connect_host:{self.connect_host}\n'
                f'connect_port:{self.connect_port}\n'
                f'region:{self.region}\n'
                f'db:{self.database}\n'
                f'user:{self.user}\n'
                f'is_iam_auth:{self.is_iam_auth}\n'
            )

            # These are synchronous boto3 HTTP calls. Run them in a worker thread
            # so a credential refresh (triggered every ``pool_expiry_min`` minutes
            # by ``check_expiry``) does not block the event loop and stall the MCP
            # stdio transport while the AWS round-trip is in flight.
            if self.is_iam_auth:
                logger.debug(f'Retrieving IAM auth token for {self.user}')
                password = await asyncio.to_thread(self.get_iam_auth_token)
            else:
                logger.debug(f'Retrieving credentials from Secrets Manager: {self.secret_arn}')
                self.user, password = await asyncio.to_thread(
                    self._get_credentials_from_secret,
                    self.secret_arn,
                    self.region,
                    self.is_test,
                )

            self.created_time = datetime.now()
            self.conninfo = f'host={self.connect_host} port={self.connect_port} dbname={self.database} user={self.user} password={password}'
            self.pool = AsyncConnectionPool(
                self.conninfo, min_size=self.min_size, max_size=self.max_size, open=False
            )

            # wait up to 30 seconds to fill the pool with connections
            try:
                await self.pool.open(True, 30)
            except Exception:
                # Pool failed to open — psycopg marks it as closed internally.
                # Set self.pool to None so callers don't try to use a closed pool.
                logger.exception('Failed to open connection pool')
                self.pool = None
                raise
            pool_name = getattr(self.pool, 'name', 'unknown')
            logger.info(
                f'Connection pool {pool_name} initialized at {self.created_time.isoformat()}, '
                f'host={self.host}, db={self.database}, is_iam={self.is_iam_auth}, '
                f'expiry_min={self.pool_expiry_min}'
            )

    async def _get_connection(self):
        """Get a database connection from the pool."""
        await self.check_expiry()

        async with self.rw_lock.reader_lock:
            if self.pool is None:
                raise ValueError('Failed to initialize connection pool')
            return self.pool.connection(timeout=15.0)

    async def check_expiry(self):
        """Check and handle pool expiry."""
        async with self.rw_lock.reader_lock:
            if self.pool and datetime.now() - self.created_time < timedelta(
                minutes=self.pool_expiry_min
            ):
                return

        pool_name = getattr(self.pool, 'name', 'None') if self.pool else 'None'
        age_seconds = (datetime.now() - self.created_time).total_seconds()
        logger.debug(
            f'check_expiry: pool {pool_name} expired or None. '
            f'age={age_seconds:.1f}s, expiry={self.pool_expiry_min * 60}s, '
            f'host={self.host}, db={self.database}'
        )
        await self.close()
        await self.initialize_pool()

    async def execute_query(
        self, sql: str, parameters: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Execute a SQL query using async connection."""
        try:
            async with await self._get_connection() as conn:
                async with conn.transaction():
                    if self.readonly_query:
                        logger.debug('SET TRANSACTION READ ONLY')
                        await conn.execute('SET TRANSACTION READ ONLY')

                    # Create a cursor for better control
                    async with conn.cursor() as cursor:
                        # Execute the query
                        if parameters:
                            converted_sql = self._convert_sql_for_psycopg(sql)
                            converted_params = self._convert_parameters(parameters)
                            await cursor.execute(converted_sql, converted_params)
                        else:
                            await cursor.execute(sql)

                        # Check if there are results to fetch by examining the cursor's description
                        if cursor.description:
                            # Get column names
                            columns = [desc[0] for desc in cursor.description]

                            # Fetch all rows
                            rows = await cursor.fetchall()

                            # Structure the response to match the interface contract required by server.py
                            column_metadata = [{'name': col} for col in columns]
                            records = []

                            # Convert each row to the expected format
                            for row in rows:
                                record = []
                                for value in row:
                                    if value is None:
                                        record.append({'isNull': True})
                                    elif isinstance(value, str):
                                        record.append({'stringValue': value})
                                    elif isinstance(value, bool):
                                        record.append({'booleanValue': value})
                                    elif isinstance(value, int):
                                        record.append({'longValue': value})
                                    elif isinstance(value, float):
                                        record.append({'doubleValue': value})
                                    elif isinstance(value, bytes):
                                        record.append({'blobValue': value})
                                    else:
                                        # Convert other types to string
                                        record.append({'stringValue': str(value)})
                                records.append(record)

                            return {'columnMetadata': column_metadata, 'records': records}
                        else:
                            # No results (e.g., for INSERT, UPDATE, etc.)
                            return {'columnMetadata': [], 'records': []}

        except Exception as e:
            # Postgres reports this as a role-level FATAL, not a distinct exception type, so
            # matching the message text is the only way to tell "role's CONNECTION LIMIT is
            # exhausted" apart from a generic connection failure and give a fixable next step
            # instead of a bare psycopg error the caller has no context to interpret.
            if 'too many connections for role' in str(e):
                logger.error(
                    f'Connection limit reached for role {self.user!r} on {self.host}: {str(e)}'
                )
                raise RuntimeError(
                    f"Connection limit reached for role '{self.user}' on {self.host}. "
                    f'Other sessions connected to this cluster are likely holding the '
                    f'remaining slots — each one is released automatically after ~10 minutes '
                    f'idle, or close one of those sessions to free a slot immediately.'
                ) from e
            logger.exception(f'Database connection error: {str(e)}')
            raise

    def _convert_parameters(self, parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Transform structured parameter format to psycopg's native parameter format."""
        result = {}
        for param in parameters:
            name = param.get('name')
            value = param.get('value', {})

            # Extract the value based on its type
            if 'stringValue' in value:
                result[name] = value['stringValue']
            elif 'longValue' in value:
                result[name] = value['longValue']
            elif 'doubleValue' in value:
                result[name] = value['doubleValue']
            elif 'booleanValue' in value:
                result[name] = value['booleanValue']
            elif 'blobValue' in value:
                result[name] = value['blobValue']
            elif 'isNull' in value and value['isNull']:
                result[name] = None

        return result

    def _convert_sql_for_psycopg(self, sql: str) -> str:
        """Convert Aurora-style :name placeholders to psycopg %(name)s style.

        Uses negative lookbehind to avoid mangling PostgreSQL's :: cast operator.

        Examples:
            :table_name     →  %(table_name)s
            column::text    →  column::text  (unchanged)
            :schema_name    →  %(schema_name)s
        """
        return re.sub(r'(?<!:):([a-zA-Z_]\w*)', r'%(\1)s', sql)

    def _get_credentials_from_secret(
        self, secret_arn: str, region: str, is_test: bool = False
    ) -> Tuple[str, str]:
        """Get database credentials from AWS Secrets Manager.

        Instance-method wrapper around the module-level
        :func:`get_credentials_from_secret` helper, kept for backward
        compatibility with existing tests.
        """
        return get_credentials_from_secret(secret_arn, region, is_test)

    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self.rw_lock.writer_lock:
            if self.pool is not None:
                pool_name = getattr(self.pool, 'name', 'unknown')
                logger.info(f'Closing connection pool {pool_name} at {datetime.now().isoformat()}')
                await self.pool.close()
                self.pool = None
                logger.info(f'Connection pool {pool_name} closed successfully')

    async def check_connection_health(self) -> bool:
        """Check if the connection is healthy."""
        try:
            result = await self.execute_query('SELECT 1')
            return len(result.get('records', [])) > 0
        except Exception as e:
            logger.exception(f'Connection health check failed: {str(e)}')
            return False

    async def get_pool_stats(self) -> Dict[str, int]:
        """Get current connection pool statistics."""
        async with self.rw_lock.reader_lock:
            if not hasattr(self, 'pool') or self.pool is None:
                return {'size': 0, 'min_size': self.min_size, 'max_size': self.max_size, 'idle': 0}

            # Access pool attributes safely
            size = getattr(self.pool, 'size', 0)
            min_size = getattr(self.pool, 'min_size', self.min_size)
            max_size = getattr(self.pool, 'max_size', self.max_size)
            idle = getattr(self.pool, 'idle', 0)

            return {'size': size, 'min_size': min_size, 'max_size': max_size, 'idle': idle}

    def get_iam_auth_token(self) -> str:
        """Generate an IAM authentication token for RDS database access."""
        rds_client = boto3.client(
            'rds', region_name=self.region, config=Config(user_agent_extra=__user_agent__)
        )
        return rds_client.generate_db_auth_token(
            DBHostname=self.host, Port=self.port, DBUsername=self.user, Region=self.region
        )
