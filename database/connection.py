"""Database connection management for the federated learning platform.

Provides both synchronous and asynchronous database session handling,
connection pooling, and initialization utilities.
"""

from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator
import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool

from database.models import Base


# Get database URLs from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/federated_medical"
)
DATABASE_ASYNC_URL = os.getenv(
    "DATABASE_ASYNC_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/federated_medical"
)


def get_engine(url: str = DATABASE_URL, echo: bool = False):
    """Create and return a SQLAlchemy engine.
    
    Args:
        url: Database connection URL
        echo: If True, log all SQL statements
        
    Returns:
        SQLAlchemy Engine instance
    """
    engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=echo,
    )
    
    # Enable foreign key constraints check on each connection
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        # For SQLite compatibility during testing
        if "sqlite" in url:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    
    return engine


def get_async_engine(url: str = DATABASE_ASYNC_URL, echo: bool = False):
    """Create and return an async SQLAlchemy engine.
    
    Args:
        url: Async database connection URL
        echo: If True, log all SQL statements
        
    Returns:
        Async SQLAlchemy Engine instance
    """
    return create_async_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=echo,
    )


# Session factories
_engine = None
_async_engine = None
_SessionLocal = None
_AsyncSessionLocal = None


def _get_session_factory() -> sessionmaker:
    """Get or create the synchronous session factory."""
    global _engine, _SessionLocal
    if _SessionLocal is None:
        _engine = get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_engine,
        )
    return _SessionLocal


def _get_async_session_factory() -> async_sessionmaker:
    """Get or create the asynchronous session factory."""
    global _async_engine, _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _async_engine = get_async_engine()
        _AsyncSessionLocal = async_sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _AsyncSessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager for database sessions.
    
    Yields:
        SQLAlchemy Session instance
        
    Example:
        with get_session() as session:
            hospital = session.query(Hospital).first()
    """
    session_factory = _get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions.
    
    Yields:
        Async SQLAlchemy Session instance
        
    Example:
        async with get_async_session() as session:
            result = await session.execute(select(Hospital))
    """
    session_factory = _get_async_session_factory()
    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI routes.
    
    Yields:
        SQLAlchemy Session instance
    """
    session_factory = _get_session_factory()
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Async dependency for FastAPI routes.
    
    Yields:
        Async SQLAlchemy Session instance
    """
    session_factory = _get_async_session_factory()
    session = session_factory()
    try:
        yield session
    finally:
        await session.close()


def init_db(drop_existing: bool = False) -> None:
    """Initialize the database by creating all tables.
    
    Args:
        drop_existing: If True, drop existing tables first
    """
    engine = get_engine()
    if drop_existing:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


async def init_async_db(drop_existing: bool = False) -> None:
    """Initialize the database asynchronously.
    
    Args:
        drop_existing: If True, drop existing tables first
    """
    engine = get_async_engine()
    async with engine.begin() as conn:
        if drop_existing:
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


def close_db() -> None:
    """Close database connections and cleanup."""
    global _engine, _async_engine, _SessionLocal, _AsyncSessionLocal
    
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionLocal = None


async def close_async_db() -> None:
    """Close async database connections and cleanup."""
    global _async_engine, _AsyncSessionLocal
    
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _AsyncSessionLocal = None
