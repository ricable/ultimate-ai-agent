#!/usr/bin/env python3
"""
Simple test for Ericsson Markdown Parser
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ericsson_markdown_parser import EricssonMarkdownParser, parse_ericsson_markdown
    print("✅ Successfully imported EricssonMarkdownParser")

    # Test with a sample file
    sample_file = Path("/Users/cedric/dev/skills/elex_features_only/en_lzn7931040_r50f_batch1/10_22104-LZA7016014_1Uen.BF.md")

    if sample_file.exists():
        print(f"📁 Testing with file: {sample_file.name}")

        try:
            feature = parse_ericsson_markdown(sample_file)
            print(f"✅ Successfully parsed feature:")
            print(f"   FAJ ID: {feature.id}")
            print(f"   Name: {feature.name[:80]}")
            print(f"   CXC Code: {feature.cxc_code}")
            print(f"   Parameters: {len(feature.parameters)}")
            print(f"   Counters: {len(feature.counters)}")

            # Show some parameters
            if feature.parameters:
                print(f"   Sample parameters:")
                for i, param in enumerate(feature.parameters[:3]):
                    print(f"     {i+1}. {param['name']} ({param['mo_class']})")

            # Show some counters
            if feature.counters:
                print(f"   Sample counters:")
                for i, counter in enumerate(feature.counters[:3]):
                    print(f"     {i+1}. {counter['name']} ({counter['category']})")

        except Exception as e:
            print(f"❌ Error parsing file: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Sample file not found: {sample_file}")

    # Test FAJ extraction patterns
    parser = EricssonMarkdownParser()
    print(f"\n🧪 Testing FAJ patterns:")
    test_cases = [
        "FAJ 121 4219",
        "FAJ1214219",
        "Feature Identity | FAJ 121 3094"
    ]

    for test in test_cases:
        faj_id = None
        for pattern in parser.faj_patterns:
            import re
            match = re.search(pattern, test)
            if match:
                if len(match.groups()) == 2:
                    faj_id = f"{match.group(1)} {match.group(2)}"
                else:
                    faj_id = match.group(1)
                    if re.match(r'\d{6}', faj_id):
                        faj_id = f"{faj_id[:3]} {faj_id[3:]}"
                break
        print(f"   '{test}' -> '{faj_id}'")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()