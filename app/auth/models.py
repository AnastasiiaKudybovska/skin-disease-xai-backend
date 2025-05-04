from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone

@dataclass
class User:
    first_name: str
    last_name: str
    username: str
    email: str
    date_of_birth: date
    password: str
    created_at: datetime = datetime.now(timezone.utc)

    def to_dict(self):
        user_dict = asdict(self)
        user_dict["date_of_birth"] = self.date_of_birth.isoformat()  
        return user_dict
