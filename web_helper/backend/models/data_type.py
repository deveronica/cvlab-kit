from sqlalchemy import Column, String, Integer
from .database import Base

class DataTypeColor(Base):
    """Stores colors for different Python data types used in the node system."""
    __tablename__ = "data_type_colors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True) # e.g., "torch.Tensor", "int"
    color = Column(String) # Hex color
