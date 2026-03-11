"""
Pipeline Engine - Orchestrates data flow through the intelligence cell
Architectural Principle: All pipelines are containerized units with standardized I/O
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import schedule
import time
from pydantic import BaseModel, Field
import firebase_admin
from firebase_admin import firestore

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution states"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class PipelineResult(BaseModel):
    """Standardized pipeline output"""
    pipeline_id: str
    execution_id: str
    status: PipelineStatus
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class BasePipeline(ABC):
    """Abstract base pipeline"""
    
    def __init__(self, pipeline_id: str, config: Dict[str, Any]):
        self.pipeline_id = pipeline_id
        self.config = config
        self.status = PipelineStatus.IDLE
        self._db = None
        self._initialize_firebase()
        
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection"""
        try:
            # Check if Firebase app already initialized
            if not firebase_admin._apps:
                cred_path = self.config.get("firebase_credentials_path")
                if cred_path:
                    cred = firebase_admin.credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred)
                else:
                    # Use GOOGLE_APPLICATION_CREDENTIALS environment variable
                    firebase_admin.initialize_app()
            
            self._db = firestore.client()
            logger.info(f"Firebase initialized for pipeline {self.pipeline_id}")
            
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            raise
    
    @abstractmethod
    async def execute(self) -> PipelineResult:
        """Execute pipeline logic"""
        pass
    
    def validate_config(self) -> bool:
        """Validate pipeline configuration"""
        required_keys = self.get_required_config_keys()
        missing = [key for key in required_keys if key not in self.config]
        
        if missing:
            logger.error(f"Missing required config keys: {missing}")
            return False
            
        return True
    
    @abstractmethod
    def get_required_config_keys(self) -> List[str]:
        """Return list of required configuration keys"""
        pass
    
    async def run_with_monitoring(self) -> PipelineResult:
        """Execute pipeline with monitoring and state persistence"""
        execution_id = f"{self.pipeline_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Update state to running
        self.status = PipelineStatus.RUNNING
        started_at = datetime.utcnow()
        
        # Persist start state
        await self._persist_state({
            "status": self.status.value,
            "started_at": started_at,
            "execution_id": execution_id
        })
        
        try:
            # Execute pipeline
            result = await self.execute()
            result.execution_id = execution_id
            result.started_at = started_at
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - started_at).total_seconds()
            result.status = PipelineStatus.COMPLETED
            
            # Update state
            self.status = PipelineStatus.COMPLETED
            
            # Persist result
            await self._persist_result(result)
            
            logger.info(f"Pipeline {self.pipeline_id} completed in {result.duration_seconds:.2f}s")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Pipeline {self.pipeline_id} failed: {error_msg}")
            
            result = PipelineResult(
                pipeline_id=self.pipeline_id,
                execution_id=execution_id,
                status=PipelineStatus.FAILED,
                error=error_msg,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.