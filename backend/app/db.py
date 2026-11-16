"""SQLite database for saved translations (history)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from signvision.config import DB_PATH

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


class Base(DeclarativeBase):
    pass


class Translation(Base):
    __tablename__ = "translations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(16))        # letters | words
    source: Mapped[str] = mapped_column(String(16))      # camera | image | video
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "mode": self.mode, "source": self.source,
                "created_at": self.created_at.isoformat(timespec="seconds")}


def init_db() -> None:
    Base.metadata.create_all(engine)


def add_translation(text: str, mode: str, source: str) -> dict:
    with Session(engine) as s:
        row = Translation(text=text, mode=mode, source=source)
        s.add(row)
        s.commit()
        return row.to_dict()


def list_translations(limit: int = 200) -> list[dict]:
    with Session(engine) as s:
        rows = s.scalars(select(Translation).order_by(Translation.id.desc()).limit(limit))
        return [r.to_dict() for r in rows]


def delete_translation(item_id: int | None = None) -> int:
    with Session(engine) as s:
        q = s.query(Translation)
        if item_id is not None:
            q = q.filter(Translation.id == item_id)
        n = q.delete()
        s.commit()
        return n
