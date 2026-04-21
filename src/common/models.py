from sqlalchemy import Column, Integer, BigInteger, String, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class News(Base):
    __tablename__ = 'news'

    message_id = Column(BigInteger, primary_key=True)
    source = Column(String, primary_key=True)
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_processed = Column(Boolean, default=False)

class ProcessingChannel(Base):
    __tablename__ = 'processing_channels'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    url = Column(String)
    status = Column(String, default='PENDING') # PENDING, SCRAPING, EXTRACTING, ACTIVE, ERROR
    progress = Column(Integer, default=0)
    error_message = Column(Text)
    last_chat_id = Column(BigInteger)
    last_message_id = Column(BigInteger)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)