from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id : Mapped[int] = mapped_column(primary_key=True)
    username : Mapped[str] = mapped_column(nullable=False)

    items : Mapped["Items"] = relationship(back_populates="owner")

class Items(Base):
    __tablename__ = "items"

    id : Mapped[int] = mapped_column(primary_key=True, index=True)
    name : Mapped[str] = mapped_column(nullable=False)
    price : Mapped[int] = mapped_column(nullable=False)

    owner : Mapped[User] = relationship(back_populates="items")