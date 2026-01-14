import requests
import json
import time
import os
import urllib.parse
from halo_paths import data_path

# Your Azure AD app details
CLIENT_ID = os.getenv('HALO_CLIENT_ID')
CLIENT_SECRET = os.getenv('HALO_CLIENT_SECRET')
REDIRECT_URI = os.getenv('HALO_REDIRECT_URI', 'http://localhost')
SCOPES = 'XboxLive.signin XboxLive.offline_access'
TOKEN_FILE = data_path("tokens.json")

def load_tokens():
    """Load saved tokens from the file"""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return json.load(f)
    return None

def save_tokens(tokens):
    """Save tokens to the file"""
    with open(TOKEN_FILE, 'w') as f:
        json.dump(tokens, f, indent=4)

def refresh_tokens(refresh_token):
    """Use the refresh token to get a new access token and user token"""
    token_resp = requests.post(
        "https://login.live.com/oauth20_token.srf",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
        }
    )
    token_resp.raise_for_status()
    tokens = token_resp.json()
    return tokens

def authenticate():
    """Authenticate the user if no tokens exist"""
    # Step 1: Build Microsoft login URL
    auth_url = (
        "https://login.live.com/oauth20_authorize.srf?"
        + urllib.parse.urlencode({
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
        })
    )

    print("🔗 Visit this URL and log in with your Microsoft account:")
    print(auth_url)
    print("\n📋 After logging in, copy the 'code' from the URL and paste it below.")
    auth_code = input("🔑 Enter authorization code: ").strip()

    # Step 2: Exchange code for access + refresh token
    token_resp = requests.post(
        "https://login.live.com/oauth20_token.srf",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": auth_code,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
        }
    )
    token_resp.raise_for_status()
    tokens = token_resp.json()

    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]
    print(f"✅ Access Token: {access_token[:30]}...")
    print(f"🔁 Refresh Token: {refresh_token[:30]}...")

    # Save tokens to the file
    save_tokens(tokens)

    return tokens

def get_user_token(access_token):
    """Get the user token and XUID"""
    user_token_resp = requests.post(
        "https://user.auth.xboxlive.com/user/authenticate",
        headers={"x-xbl-contract-version": "1"},
        json={
            "Properties": {
                "AuthMethod": "RPS",
                "SiteName": "user.auth.xboxlive.com",
                "RpsTicket": f"d={access_token}"
            },
            "RelyingParty": "http://auth.xboxlive.com",
            "TokenType": "JWT"
        }
    )
    user_token_resp.raise_for_status()
    user_data = user_token_resp.json()
    user_token = user_data["Token"]
    user_hash = user_data["DisplayClaims"]["xui"][0]["uhs"]
    
    # Get XUID from the response
    xui_data = user_data["DisplayClaims"]["xui"][0]
    print("Available keys in xui data:", xui_data.keys())
    xuid = None
    if "xid" in xui_data:
        xuid = xui_data["xid"]
    elif "id" in xui_data:
        xuid = xui_data["id"]
    else:
        print("⚠️ Could not find XUID in user data")
        print("📌 Using provided XUID: 2535417291060944")
        xuid = "2535417291060944"  # Using the provided XUID
        
    print(f"✅ User Hash: {user_hash}")
    if xuid:
        print(f"✅ XUID: {xuid}")
    
    return user_token, xuid

def get_xsts_token(user_token):
    """Get the XSTS token"""
    xsts_resp = requests.post(
        "https://xsts.auth.xboxlive.com/xsts/authorize",
        headers={"x-xbl-contract-version": "1"},
        json={
            "Properties": {
                "SandboxId": "RETAIL",
                "UserTokens": [user_token]
            },
            "RelyingParty": "https://prod.xsts.halowaypoint.com/",
            "TokenType": "JWT"
        }
    )
    xsts_resp.raise_for_status()
    xsts_data = xsts_resp.json()
    xsts_token = xsts_data["Token"]
    print(f"✅ XSTS Token: {xsts_token[:30]}...")
    return xsts_token

def get_spartan_token(xsts_token):
    """Get the Spartan v4 token"""
    spartan_url = "https://settings.svc.halowaypoint.com/spartan-token"
    headers = {
        "User-Agent": "HaloInfinite/6.10022.0.0 (Windows;10;;Professional, x64)",
        "Accept": "application/json"
    }
    payload = {
        "Audience": "urn:343:s3:services",
        "MinVersion": "4",
        "Proof": [
            {
                "Token": xsts_token,
                "TokenType": "Xbox_XSTSv3"
            }
        ]
    }
    spartan_resp = requests.post(spartan_url, headers=headers, json=payload)
    if not spartan_resp.ok:
        print(f"❌ Spartan Token Request Failed: {spartan_resp.status_code}")
        print(spartan_resp.text)
        return None

    spartan_data = spartan_resp.json()
    spartan_token = spartan_data["SpartanToken"]
    print(f"✅ Spartan Token: {spartan_token[:30]}...")
    return spartan_token

def get_clearance_token(spartan_token, xuid):
    """Get the clearance token"""
    if not xuid:
        print("❌ XUID is missing, cannot get clearance token")
        return None
        
    # Ensure XUID is a string
    xuid = str(xuid)
        
    clearance_url = f"https://settings.svc.halowaypoint.com/oban/flight-configurations/titles/hi/audiences/RETAIL/players/xuid({xuid})/active"
    
    headers = {
        "User-Agent": "HaloInfinite/6.10022.0.0 (Windows;10;;Professional, x64)",
        "Accept": "application/json",
        "x-343-authorization-spartan": spartan_token
    }
    
    params = {
        "sandbox": "UNUSED",
        "build": "210921.22.01.10.1706-0"
    }
    
    try:
        print(f"🔍 Requesting clearance token from: {clearance_url}")
        print(f"🔐 Using headers: {headers}")
        print(f"🔍 Using params: {params}")
        
        clearance_resp = requests.get(clearance_url, headers=headers, params=params)
        
        print(f"📋 Response status code: {clearance_resp.status_code}")
        print(f"📋 Response headers: {dict(clearance_resp.headers)}")
        
        if not clearance_resp.ok:
            print(f"❌ Clearance Token Request Failed: {clearance_resp.status_code}")
            print(clearance_resp.text)
            return None
        
        try:
            clearance_data = clearance_resp.json()
            print(f"📋 Response data: {json.dumps(clearance_data, indent=2)}")
            
            clearance_token = clearance_data.get("FlightConfigurationId")
            if clearance_token:
                print(f"✅ Clearance Token: {clearance_token}")
                return clearance_token
            else:
                print("❌ Clearance token not found in response")
                print(json.dumps(clearance_data, indent=2))
                return None
        except json.JSONDecodeError:
            print("❌ Could not parse JSON response")
            print("📋 Raw response:", clearance_resp.text)
            return None
            
    except Exception as e:
        print(f"❌ Error getting clearance token: {str(e)}")
        return None

def main():
    """Main entry point"""
    tokens = load_tokens()
    if tokens:
        # Check if the token is expired
        expiration_time = tokens.get("expires_at", 0)
        current_time = time.time()
        if current_time > expiration_time:
            print("❌ Tokens have expired, refreshing...")
            tokens = refresh_tokens(tokens["refresh_token"])
        else:
            print("✅ Using saved tokens...")

        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]
    else:
        # If no tokens saved, start authentication
        print("🔑 Starting authentication...")
        tokens = authenticate()
        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]

    # Get User Token and XUID
    user_token, xuid = get_user_token(access_token)
    
    # Store user token and XUID in tokens dictionary
    tokens["user_token"] = user_token
    if xuid:
        tokens["xuid"] = xuid
    
    # Get XSTS Token
    xsts_token = get_xsts_token(user_token)
    tokens["xsts_token"] = xsts_token
    
    # Get Spartan Token
    spartan_token = get_spartan_token(xsts_token)
    tokens["spartan_token"] = spartan_token
    
    # Get Clearance Token
    if spartan_token and xuid:
        clearance_token = get_clearance_token(spartan_token, xuid)
        if clearance_token:
            tokens["clearance_token"] = clearance_token
    
    # Save the updated tokens with expiration time
    tokens["expires_at"] = time.time() + tokens.get("expires_in", 3600)  # Set expiration time
    save_tokens(tokens)
    
    # Print summary of obtained tokens
    print("\n📝 Token Summary:")
    for token_type in ["access_token", "refresh_token", "user_token", "xuid", "xsts_token", "spartan_token", "clearance_token"]:
        if token_type in tokens and tokens[token_type]:
            value = tokens[token_type]
            if isinstance(value, str) and len(value) > 20:
                value = f"{value[:20]}..."
            print(f"  ✅ {token_type}: {value}")
        else:
            print(f"  ❌ {token_type}: Not available")
    
    print(f"\n✅ All tokens have been saved to {TOKEN_FILE}")

if __name__ == "__main__":
    main()
