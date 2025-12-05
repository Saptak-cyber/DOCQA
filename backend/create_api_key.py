"""
Utility script to create and manage API keys for tenants.
Usage:
    python create_api_key.py --tenant my-tenant --name "Production Key"
"""

import sys
import argparse
from app.database import get_db_context
from app.auth import create_api_key_for_tenant


def main():
    parser = argparse.ArgumentParser(description="Create API key for a tenant")
    parser.add_argument("--tenant", required=True, help="Tenant ID")
    parser.add_argument("--name", help="Friendly name for the key")
    parser.add_argument("--rate-minute", type=int, default=10, help="Rate limit per minute")
    parser.add_argument("--rate-hour", type=int, default=100, help="Rate limit per hour")
    
    args = parser.parse_args()
    
    with get_db_context() as db:
        try:
            plain_key, key_record = create_api_key_for_tenant(
                db=db,
                tenant_id=args.tenant,
                name=args.name,
                rate_limit_minute=args.rate_minute,
                rate_limit_hour=args.rate_hour
            )
            
            print("\n" + "="*60)
            print("✓ API Key Created Successfully!")
            print("="*60)
            print(f"\nTenant ID:  {key_record.tenant_id}")
            print(f"Key ID:     {key_record.key_id}")
            print(f"Name:       {key_record.name or '(none)'}")
            print(f"\n{'='*60}")
            print(f"API Key:    {plain_key}")
            print("="*60)
            print("\n⚠️  IMPORTANT: Save this key securely!")
            print("   This is the ONLY time you will see it.\n")
            
        except Exception as e:
            print(f"\n✗ Error creating API key: {e}\n", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
