"""
AGRICONNECT - Database Models
User Module:         User table
Marketplace Module:  Listing table, ContactRequest table
Advisory Module:      AdvisoryRule table
Subsidy Module:        SubsidyRule table
Price Module:          PriceHistory table
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(10), nullable=False)   # 'farmer' or 'buyer'

    # --- login attempt limiting ---
    failed_attempts = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f"<User {self.name} ({self.role})>"


class Listing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    crop_name = db.Column(db.String(80), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), nullable=False, default='kg')
    price_per_unit = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(120), nullable=False)
    description = db.Column(db.String(300), nullable=True)

    date_posted = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    farmer = db.relationship('User', backref='listings')

    def __repr__(self):
        return f"<Listing {self.crop_name} by {self.farmer_id}>"


class ContactRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    listing_id = db.Column(db.Integer, db.ForeignKey('listing.id'), nullable=False)
    buyer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(300), nullable=True)
    date_sent = db.Column(db.DateTime, default=datetime.utcnow)

    listing = db.relationship('Listing', backref='contact_requests')
    buyer = db.relationship('User', backref='sent_requests')

    def __repr__(self):
        return f"<ContactRequest listing={self.listing_id} buyer={self.buyer_id}>"


class AdvisoryRule(db.Model):
    """
    A single row of the rule-based Crop Advisory engine.
    crop + symptom (+ optional soil) -> issue + recommendation.
    soil = 'Any' means this rule applies regardless of soil condition.
    """
    id = db.Column(db.Integer, primary_key=True)
    crop = db.Column(db.String(50), nullable=False)
    soil = db.Column(db.String(30), nullable=False, default='Any')
    symptom = db.Column(db.String(80), nullable=False)
    issue = db.Column(db.String(120), nullable=False)
    recommendation = db.Column(db.String(300), nullable=False)

    def __repr__(self):
        return f"<AdvisoryRule {self.crop}/{self.soil}/{self.symptom} -> {self.issue}>"


class SubsidyRule(db.Model):
    """
    A single government scheme entry for the Subsidy Eligibility Module.
    region / crop / land_category = 'Any' means that field doesn't restrict eligibility.
    A farmer's (region, crop, land_category) input can match MULTIPLE schemes at once,
    unlike AdvisoryRule which returns a single best match.
    """
    id = db.Column(db.Integer, primary_key=True)
    scheme_name = db.Column(db.String(120), nullable=False)
    region = db.Column(db.String(50), nullable=False, default='Any')
    crop = db.Column(db.String(50), nullable=False, default='Any')
    land_category = db.Column(db.String(30), nullable=False, default='Any')
    benefit = db.Column(db.String(300), nullable=False)
    how_to_apply = db.Column(db.String(250), nullable=False)

    def __repr__(self):
        return f"<SubsidyRule {self.scheme_name} ({self.region}/{self.crop}/{self.land_category})>"


class PriceHistory(db.Model):
    """
    One weekly average mandi price for a (region, crop) pair, in Rs per quintal.
    Feeds the Price Predictor module. source = 'demo' (synthetic) or 'agmarknet' (real).
    """
    id = db.Column(db.Integer, primary_key=True)
    week_end = db.Column(db.Date, nullable=False, index=True)   # Sunday that ends the week
    region = db.Column(db.String(50), nullable=False, index=True)
    crop = db.Column(db.String(50), nullable=False, index=True)
    modal_price = db.Column(db.Float, nullable=False)           # Rs / quintal
    source = db.Column(db.String(20), nullable=False, default='agmarknet')

    def __repr__(self):
        return f"<PriceHistory {self.crop}/{self.region} {self.week_end} Rs{self.modal_price}>"
