from app import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    age = db.Column(db.Integer, unique=False, nullable=False)
    job = db.Column(db.Integer, unique=False, nullable=False)
    marital = db.Column(db.Integer, unique=False, nullable=False)
    education = db.Column(db.Integer, unique=False, nullable=False)
    default = db.Column(db.Integer, unique=False, nullable=False)
    balance = db.Column(db.Integer, unique=False, nullable=False)
    housing = db.Column(db.Integer, unique=False, nullable=False)
    loan = db.Column(db.Integer, unique=False, nullable=False)
    contact = db.Column(db.Integer, unique=False, nullable=False)
    day = db.Column(db.Integer, unique=False, nullable=False)
    month = db.Column(db.Integer, unique=False, nullable=False)
    campaign = db.Column(db.Integer, unique=False, nullable=False)
    pdays = db.Column(db.Integer, unique=False, nullable=False)
    previous = db.Column(db.Integer, unique=False, nullable=False)
    poutcome = db.Column(db.Integer, unique=False, nullable=False)
    y = db.Column(db.Integer, unique=False, nullable=False)

    def __repr__(self):
        return f"User('{self.id}', '{self.y}')"

