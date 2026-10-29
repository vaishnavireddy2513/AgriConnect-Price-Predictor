"""
AGRICONNECT - Main Flask Application
User Module:        Signup, Login, Logout, Dashboard, Edit Profile, Change Password
Marketplace Module: Add/Edit/Delete Listing, Browse/Search Listings, Contact Requests
Advisory Module:    Rule-based crop/soil/symptom advisory
Subsidy Module:     Rule-based region/crop/land-size scheme eligibility
Price Module:       ML price forecast (next 1-4 weeks) per crop and region
"""

from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Listing, ContactRequest, AdvisoryRule, SubsidyRule, PriceHistory
from advisory_data import CROPS, SOILS, SYMPTOMS, BASE_RULES, SOIL_OVERRIDES
from subsidy_data import REGIONS, LAND_CATEGORIES, SCHEME_RULES
from price_engine import (PRICE_CROPS, PRICE_REGIONS, ALL_REGIONS, MAX_HORIZON,
                          seed_price_history, ensure_price_model,
                          get_forecast, get_listing_insight)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-key-later'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agriconnect.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=14)

db.init_app(app)


def seed_advisory_rules():
    """Populate the AdvisoryRule table from advisory_data.py, only if it's empty."""
    if AdvisoryRule.query.first() is not None:
        return

    rows = []
    for crop in CROPS:
        for symptom in SYMPTOMS:
            issue, reco_template = BASE_RULES[symptom]
            rows.append(AdvisoryRule(
                crop=crop, soil='Any', symptom=symptom,
                issue=issue, recommendation=reco_template.format(crop=crop)
            ))
        for (sym, soil_key), (issue, reco_template) in SOIL_OVERRIDES.items():
            rows.append(AdvisoryRule(
                crop=crop, soil=soil_key, symptom=sym,
                issue=issue, recommendation=reco_template.format(crop=crop)
            ))
    db.session.bulk_save_objects(rows)
    db.session.commit()


def seed_subsidy_rules():
    """Populate the SubsidyRule table from subsidy_data.py, only if it's empty."""
    if SubsidyRule.query.first() is not None:
        return

    rows = []
    for scheme_name, region, crop, land_category, benefit, how_to_apply in SCHEME_RULES:
        rows.append(SubsidyRule(
            scheme_name=scheme_name, region=region, crop=crop, land_category=land_category,
            benefit=benefit, how_to_apply=how_to_apply
        ))
    db.session.bulk_save_objects(rows)
    db.session.commit()


with app.app_context():
    db.create_all()
    seed_advisory_rules()
    seed_subsidy_rules()
    seed_price_history()      # demo data on first run; replace with real data via manage_prices.py
    ensure_price_model()      # trains the price model on first run

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
UNITS = ['kg', 'quintal']


# ============================================================
# HELPERS
# ============================================================
def is_valid_phone(phone):
    return phone.isdigit() and len(phone) == 10


def login_required():
    return 'user_id' in session


def current_user():
    if 'user_id' not in session:
        return None
    return User.query.get(session['user_id'])


# ============================================================
# USER MODULE
# ============================================================
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '')
        role = request.form.get('role', '')

        if not name or not phone or not password or not role:
            flash("Please fill in every field before continuing.", "error")
            return render_template('signup.html')

        if not is_valid_phone(phone):
            flash("Phone number must be exactly 10 digits.", "error")
            return render_template('signup.html')

        if User.query.filter_by(phone=phone).first():
            flash("An account with this phone number already exists. Please log in instead.", "error")
            return render_template('signup.html')

        new_user = User(
            name=name, phone=phone,
            password_hash=generate_password_hash(password),
            role=role
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '')
        remember_me = request.form.get('remember_me')

        user = User.query.filter_by(phone=phone).first()

        if user and user.locked_until and user.locked_until > datetime.utcnow():
            minutes_left = int((user.locked_until - datetime.utcnow()).total_seconds() // 60) + 1
            flash(f"Too many failed attempts. This account is locked for {minutes_left} more minute(s).", "error")
            return render_template('login.html')

        if user and check_password_hash(user.password_hash, password):
            user.failed_attempts = 0
            user.locked_until = None
            db.session.commit()

            session.permanent = bool(remember_me)
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_role'] = user.role

            flash(f"Welcome back, {user.name}!", "success")
            return redirect(url_for('dashboard'))
        else:
            if user:
                user.failed_attempts += 1
                if user.failed_attempts >= MAX_LOGIN_ATTEMPTS:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_MINUTES)
                    db.session.commit()
                    flash(f"Too many failed attempts. This account is now locked for {LOCKOUT_MINUTES} minutes.", "error")
                    return render_template('login.html')
                else:
                    remaining = MAX_LOGIN_ATTEMPTS - user.failed_attempts
                    db.session.commit()
                    flash(f"Incorrect phone number or password. {remaining} attempt(s) remaining before lockout.", "error")
                    return render_template('login.html')

            flash("Incorrect phone number or password. Please try again.", "error")
            return render_template('login.html')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))
    return render_template('dashboard.html', name=session['user_name'], role=session['user_role'])


@app.route('/profile/edit', methods=['GET', 'POST'])
def edit_profile():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    user = current_user()

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        phone = request.form.get('phone', '').strip()

        if not name or not phone:
            flash("Name and phone number cannot be empty.", "error")
            return render_template('edit_profile.html', user=user)

        if not is_valid_phone(phone):
            flash("Phone number must be exactly 10 digits.", "error")
            return render_template('edit_profile.html', user=user)

        existing = User.query.filter(User.phone == phone, User.id != user.id).first()
        if existing:
            flash("That phone number is already linked to another account.", "error")
            return render_template('edit_profile.html', user=user)

        user.name = name
        user.phone = phone
        db.session.commit()

        session['user_name'] = user.name
        flash("Profile updated successfully.", "success")
        return redirect(url_for('dashboard'))

    return render_template('edit_profile.html', user=user)


@app.route('/change-password', methods=['GET', 'POST'])
def change_password():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        user = current_user()

        if not check_password_hash(user.password_hash, current_password):
            flash("Your current password is incorrect.", "error")
            return render_template('change_password.html')

        if len(new_password) < 6:
            flash("New password must be at least 6 characters long.", "error")
            return render_template('change_password.html')

        if new_password != confirm_password:
            flash("New password and confirmation do not match.", "error")
            return render_template('change_password.html')

        user.password_hash = generate_password_hash(new_password)
        db.session.commit()

        session.clear()
        flash("Password changed successfully. Please log in again.", "success")
        return redirect(url_for('login'))

    return render_template('change_password.html')


# ============================================================
# MARKETPLACE MODULE
# ============================================================

@app.route('/marketplace')
def marketplace():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    crop_query = request.args.get('crop', '').strip()
    location_query = request.args.get('location', '').strip()

    query = Listing.query.filter_by(is_active=True)
    if crop_query:
        query = query.filter(Listing.crop_name.ilike(f'%{crop_query}%'))
    if location_query:
        query = query.filter(Listing.location.ilike(f'%{location_query}%'))

    listings = query.order_by(Listing.date_posted.desc()).all()

    return render_template('marketplace.html', listings=listings,
                            crop_query=crop_query, location_query=location_query)


@app.route('/marketplace/listing/<int:listing_id>')
def listing_detail(listing_id):
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    listing = Listing.query.get_or_404(listing_id)
    already_contacted = False
    if session['user_role'] == 'buyer':
        already_contacted = ContactRequest.query.filter_by(
            listing_id=listing.id, buyer_id=session['user_id']
        ).first() is not None

    insight = get_listing_insight(listing)
    return render_template('listing_detail.html', listing=listing,
                           already_contacted=already_contacted, insight=insight)


@app.route('/marketplace/add', methods=['GET', 'POST'])
def add_listing():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    if session['user_role'] != 'farmer':
        flash("Only farmers can list produce.", "error")
        return redirect(url_for('marketplace'))

    if request.method == 'POST':
        crop_name = request.form.get('crop_name', '').strip()
        quantity = request.form.get('quantity', '').strip()
        unit = request.form.get('unit', 'kg')
        price_per_unit = request.form.get('price_per_unit', '').strip()
        location = request.form.get('location', '').strip()
        description = request.form.get('description', '').strip()

        if not crop_name or not quantity or not price_per_unit or not location:
            flash("Please fill in crop, quantity, price, and location.", "error")
            return render_template('add_listing.html', units=UNITS)

        try:
            quantity_val = float(quantity)
            price_val = float(price_per_unit)
            if quantity_val <= 0 or price_val <= 0:
                raise ValueError
        except ValueError:
            flash("Quantity and price must be positive numbers.", "error")
            return render_template('add_listing.html', units=UNITS)

        if unit not in UNITS:
            unit = 'kg'

        new_listing = Listing(
            farmer_id=session['user_id'],
            crop_name=crop_name,
            quantity=quantity_val,
            unit=unit,
            price_per_unit=price_val,
            location=location,
            description=description or None
        )
        db.session.add(new_listing)
        db.session.commit()

        flash("Your produce has been listed successfully.", "success")
        return redirect(url_for('my_listings'))

    return render_template('add_listing.html', units=UNITS)


@app.route('/marketplace/my-listings')
def my_listings():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    if session['user_role'] != 'farmer':
        flash("Only farmers have listings to manage.", "error")
        return redirect(url_for('marketplace'))

    listings = Listing.query.filter_by(farmer_id=session['user_id']).order_by(Listing.date_posted.desc()).all()
    return render_template('my_listings.html', listings=listings)


@app.route('/marketplace/listing/<int:listing_id>/edit', methods=['GET', 'POST'])
def edit_listing(listing_id):
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    listing = Listing.query.get_or_404(listing_id)
    if listing.farmer_id != session['user_id']:
        flash("You can only edit your own listings.", "error")
        return redirect(url_for('my_listings'))

    if request.method == 'POST':
        crop_name = request.form.get('crop_name', '').strip()
        quantity = request.form.get('quantity', '').strip()
        unit = request.form.get('unit', 'kg')
        price_per_unit = request.form.get('price_per_unit', '').strip()
        location = request.form.get('location', '').strip()
        description = request.form.get('description', '').strip()

        if not crop_name or not quantity or not price_per_unit or not location:
            flash("Please fill in crop, quantity, price, and location.", "error")
            return render_template('edit_listing.html', listing=listing, units=UNITS)

        try:
            quantity_val = float(quantity)
            price_val = float(price_per_unit)
            if quantity_val <= 0 or price_val <= 0:
                raise ValueError
        except ValueError:
            flash("Quantity and price must be positive numbers.", "error")
            return render_template('edit_listing.html', listing=listing, units=UNITS)

        listing.crop_name = crop_name
        listing.quantity = quantity_val
        listing.unit = unit if unit in UNITS else 'kg'
        listing.price_per_unit = price_val
        listing.location = location
        listing.description = description or None
        db.session.commit()

        flash("Listing updated successfully.", "success")
        return redirect(url_for('my_listings'))

    return render_template('edit_listing.html', listing=listing, units=UNITS)


@app.route('/marketplace/listing/<int:listing_id>/mark-sold', methods=['POST'])
def mark_sold(listing_id):
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    listing = Listing.query.get_or_404(listing_id)
    if listing.farmer_id != session['user_id']:
        flash("You can only update your own listings.", "error")
        return redirect(url_for('my_listings'))

    listing.is_active = False
    db.session.commit()
    flash(f"'{listing.crop_name}' marked as sold and removed from Marketplace.", "success")
    return redirect(url_for('my_listings'))


@app.route('/marketplace/listing/<int:listing_id>/delete', methods=['POST'])
def delete_listing(listing_id):
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    listing = Listing.query.get_or_404(listing_id)
    if listing.farmer_id != session['user_id']:
        flash("You can only delete your own listings.", "error")
        return redirect(url_for('my_listings'))

    ContactRequest.query.filter_by(listing_id=listing.id).delete()
    db.session.delete(listing)
    db.session.commit()
    flash("Listing deleted.", "success")
    return redirect(url_for('my_listings'))


@app.route('/marketplace/listing/<int:listing_id>/contact', methods=['POST'])
def contact_farmer(listing_id):
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    if session['user_role'] != 'buyer':
        flash("Only buyers can send contact requests.", "error")
        return redirect(url_for('listing_detail', listing_id=listing_id))

    listing = Listing.query.get_or_404(listing_id)
    message = request.form.get('message', '').strip()

    existing = ContactRequest.query.filter_by(listing_id=listing.id, buyer_id=session['user_id']).first()
    if existing:
        flash("You've already sent an interest request for this listing.", "error")
        return redirect(url_for('listing_detail', listing_id=listing_id))

    new_request = ContactRequest(
        listing_id=listing.id,
        buyer_id=session['user_id'],
        message=message or None
    )
    db.session.add(new_request)
    db.session.commit()

    flash(f"Interest sent! Farmer's contact number: {listing.farmer.phone}", "success")
    return redirect(url_for('listing_detail', listing_id=listing_id))


@app.route('/marketplace/my-interests')
def my_interests():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    if session['user_role'] != 'buyer':
        flash("Only buyers have sent interest requests.", "error")
        return redirect(url_for('marketplace'))

    requests_sent = ContactRequest.query.filter_by(buyer_id=session['user_id']).order_by(ContactRequest.date_sent.desc()).all()
    return render_template('my_interests.html', requests_sent=requests_sent)


@app.route('/marketplace/received-interests')
def received_interests():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    if session['user_role'] != 'farmer':
        flash("Only farmers receive interest requests.", "error")
        return redirect(url_for('marketplace'))

    my_listing_ids = [l.id for l in Listing.query.filter_by(farmer_id=session['user_id']).all()]
    requests_received = ContactRequest.query.filter(
        ContactRequest.listing_id.in_(my_listing_ids)
    ).order_by(ContactRequest.date_sent.desc()).all()

    return render_template('received_interests.html', requests_received=requests_received)


# ============================================================
# CROP ADVISORY MODULE
# ============================================================

def get_advisory(crop, soil, symptom):
    """
    Rule-based decision function. Isolated in its own function so it can be
    swapped for a trained ML classifier later without changing the route,
    the form, or the database structure around it.

    Matching priority:
      1. Exact match on (crop, soil, symptom)   -> most specific
      2. Fallback match on (crop, 'Any', symptom) -> general rule for that crop
      3. None if nothing matches
    """
    rule = AdvisoryRule.query.filter_by(crop=crop, soil=soil, symptom=symptom).first()
    if rule is None:
        rule = AdvisoryRule.query.filter_by(crop=crop, soil='Any', symptom=symptom).first()
    return rule


@app.route('/advisory', methods=['GET', 'POST'])
def advisory():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    result = None
    selected = {'crop': CROPS[0], 'soil': SOILS[0], 'symptom': SYMPTOMS[0]}

    if request.method == 'POST':
        crop = request.form.get('crop', '')
        soil = request.form.get('soil', '')
        symptom = request.form.get('symptom', '')
        selected = {'crop': crop, 'soil': soil, 'symptom': symptom}

        if crop not in CROPS or soil not in SOILS or symptom not in SYMPTOMS:
            flash("Please select a valid crop, soil condition, and symptom.", "error")
        else:
            result = get_advisory(crop, soil, symptom)
            if result is None:
                flash("No specific guidance found for this combination. Please consult your "
                      "local agriculture officer.", "error")

    return render_template('advisory.html', crops=CROPS, soils=SOILS, symptoms=SYMPTOMS,
                            selected=selected, result=result)


# ============================================================
# SUBSIDY ELIGIBILITY MODULE
# ============================================================

def get_subsidies(region, crop, land_category):
    """
    Rule-based multi-match lookup. Unlike get_advisory(), this can return
    several matching schemes at once, since a farmer is usually eligible
    for more than one government scheme simultaneously.

    A scheme matches if, for each of region/crop/land_category, the scheme's
    stored value is either 'Any' (no restriction) or exactly equal to the
    farmer's selected value.
    """
    all_schemes = SubsidyRule.query.all()
    matches = []
    for scheme in all_schemes:
        region_ok = scheme.region == 'Any' or scheme.region == region
        crop_ok = scheme.crop == 'Any' or scheme.crop == crop
        land_ok = scheme.land_category == 'Any' or scheme.land_category == land_category
        if region_ok and crop_ok and land_ok:
            matches.append(scheme)
    return matches


@app.route('/subsidy', methods=['GET', 'POST'])
def subsidy():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    results = None
    selected = {'region': REGIONS[0], 'crop': CROPS[0], 'land_category': LAND_CATEGORIES[0]}

    if request.method == 'POST':
        region = request.form.get('region', '')
        crop = request.form.get('crop', '')
        land_category = request.form.get('land_category', '')
        selected = {'region': region, 'crop': crop, 'land_category': land_category}

        if region not in REGIONS or crop not in CROPS or land_category not in LAND_CATEGORIES:
            flash("Please select a valid region, crop, and land size.", "error")
        else:
            results = get_subsidies(region, crop, land_category)
            if len(results) == 0:
                flash("No matching schemes found for this combination. Please check with your "
                      "local Agriculture Department for the latest scheme information.", "error")

    return render_template('subsidy.html', regions=REGIONS, crops=CROPS,
                            land_categories=LAND_CATEGORIES, selected=selected, results=results)


# ============================================================
# PRICE PREDICTOR MODULE
# ============================================================

@app.route('/price-predictor', methods=['GET', 'POST'])
def price_predictor():
    if not login_required():
        flash("Please log in to continue.", "error")
        return redirect(url_for('login'))

    regions = [ALL_REGIONS] + PRICE_REGIONS
    result = None
    selected = {'crop': PRICE_CROPS[0], 'region': ALL_REGIONS, 'weeks': MAX_HORIZON}

    if request.method == 'POST':
        crop = request.form.get('crop', '')
        region = request.form.get('region', '')
        try:
            weeks = int(request.form.get('weeks', MAX_HORIZON))
        except ValueError:
            weeks = MAX_HORIZON
        selected = {'crop': crop, 'region': region, 'weeks': weeks}

        if crop not in PRICE_CROPS or region not in regions or not 1 <= weeks <= MAX_HORIZON:
            flash("Please select a valid crop, region, and forecast length.", "error")
        else:
            result = get_forecast(crop, region, weeks)
            if result is None:
                flash("Not enough price data for this combination yet. Please try another.", "error")

    return render_template('price_predictor.html', crops=PRICE_CROPS, regions=regions,
                           week_options=list(range(1, MAX_HORIZON + 1)),
                           selected=selected, result=result)


if __name__ == '__main__':
    app.run(debug=True)
