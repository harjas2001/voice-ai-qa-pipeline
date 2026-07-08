"""
Conversation ID Filter and Extraction Tool

Extracts rows from a master data file based on a list of conversation IDs.
Handles multiple rows per conversation ID and reports missing IDs.

Usage:
    python extract_by_conversation_id.py master_data.xlsx conversation_ids.csv output.csv
    python extract_by_conversation_id.py master_data.csv conversation_ids.csv output.csv

Features:
    - Supports Excel (.xlsx) and CSV master files
    - Extracts ALL rows matching each conversation ID
    - Reports conversation IDs not found in master data
    - Handles duplicates and multiple rows per conversation

Author: Harjas @ TPG Telecom
"""

import pandas as pd
import sys
import argparse
from pathlib import Path


def load_master_data(filepath):
    """
    Load master data from Excel or CSV file
    
    Args:
        filepath: Path to master data file (.xlsx or .csv)
        
    Returns:
        DataFrame with master data
    """
    print(f"\n📂 Loading master data: {filepath}")
    
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Master data file not found: {filepath}")
    
    # Determine file type and load
    if file_path.suffix.lower() in ['.xlsx', '.xls']:
        try:
            df = pd.read_excel(filepath)
            print(f"   ✅ Loaded Excel file")
        except Exception as e:
            raise Exception(f"Error reading Excel file: {e}")
    elif file_path.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            print(f"   ✅ Loaded CSV file")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
            print(f"   ✅ Loaded CSV file (ISO-8859-1 encoding)")
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}. Use .xlsx or .csv")
    
    print(f"   📊 Master data: {len(df):,} total rows")
    print(f"   📋 Columns: {len(df.columns)}")
    
    # Check for conversation_id column
    if 'conversation_id' not in df.columns:
        print(f"\n❌ Error: Master data must have 'conversation_id' column")
        print(f"   Available columns: {', '.join(df.columns.tolist())}")
        raise ValueError("Missing 'conversation_id' column in master data")
    
    # Show unique conversation IDs
    unique_convos = df['conversation_id'].nunique()
    print(f"   🔑 Unique conversation IDs: {unique_convos:,}")
    
    return df


def load_conversation_ids(filepath):
    """
    Load conversation IDs from CSV file
    
    Args:
        filepath: Path to CSV file with conversation IDs
        
    Returns:
        Set of conversation IDs
    """
    print(f"\n📂 Loading conversation ID filter: {filepath}")
    
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Conversation ID file not found: {filepath}")
    
    # Load CSV
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    
    # Get the first column (should be conversation IDs)
    first_column = df.columns[0]
    conversation_ids = df[first_column].dropna().unique().tolist()
    
    # Convert to strings and remove any whitespace
    conversation_ids = [str(cid).strip() for cid in conversation_ids]
    
    print(f"   ✅ Loaded {len(conversation_ids):,} unique conversation IDs")
    print(f"   📋 Column name: '{first_column}'")
    
    return set(conversation_ids), first_column


def extract_matching_rows(master_df, conversation_ids):
    """
    Extract all rows from master data that match the conversation IDs
    
    Args:
        master_df: Master DataFrame
        conversation_ids: Set of conversation IDs to filter
        
    Returns:
        Filtered DataFrame and statistics
    """
    print(f"\n🔍 Filtering master data by conversation IDs...")
    
    # Convert master conversation_id to string for matching
    master_df['conversation_id'] = master_df['conversation_id'].astype(str).str.strip()
    
    # Filter rows where conversation_id matches
    filtered_df = master_df[master_df['conversation_id'].isin(conversation_ids)].copy()
    
    # Get statistics
    total_rows = len(filtered_df)
    matched_conversation_ids = filtered_df['conversation_id'].unique().tolist()
    num_matched = len(matched_conversation_ids)
    num_requested = len(conversation_ids)
    num_not_found = num_requested - num_matched
    
    print(f"   ✅ Extracted {total_rows:,} rows")
    print(f"   ✅ Matched {num_matched:,} conversation IDs (out of {num_requested:,} requested)")
    
    if num_not_found > 0:
        print(f"   ⚠️  {num_not_found:,} conversation IDs NOT found in master data")
    
    # Find which conversation IDs were not found
    not_found_ids = conversation_ids - set(matched_conversation_ids)
    
    # Show row distribution
    rows_per_convo = filtered_df.groupby('conversation_id').size()
    print(f"\n📊 Row Distribution:")
    print(f"   Min rows per conversation: {rows_per_convo.min()}")
    print(f"   Max rows per conversation: {rows_per_convo.max()}")
    print(f"   Average rows per conversation: {rows_per_convo.mean():.1f}")
    
    return filtered_df, num_not_found, not_found_ids


def save_output(df, output_filepath, not_found_count, not_found_ids, save_missing=False):
    """
    Save filtered data to CSV
    
    Args:
        df: Filtered DataFrame
        output_filepath: Path for output CSV
        not_found_count: Number of conversation IDs not found
        not_found_ids: Set of conversation IDs not found
        save_missing: Whether to save missing IDs to a separate file
    """
    print(f"\n💾 Saving output: {output_filepath}")
    
    # Save main output
    df.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f"   ✅ Saved {len(df):,} rows")
    
    # Optionally save missing conversation IDs
    if save_missing and not_found_ids:
        missing_filepath = output_filepath.replace('.csv', '_missing_ids.csv')
        missing_df = pd.DataFrame({
            'conversation_id': sorted(list(not_found_ids))
        })
        missing_df.to_csv(missing_filepath, index=False, encoding='utf-8')
        print(f"   📄 Saved {len(not_found_ids):,} missing IDs to: {missing_filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract rows from master data based on conversation IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python extract_by_conversation_id.py master_data.xlsx conversation_ids.csv output.csv
  
  # With missing IDs report
  python extract_by_conversation_id.py master_data.csv ids.csv output.csv --save-missing
  
  # Show sample output
  python extract_by_conversation_id.py master_data.xlsx ids.csv output.csv --show-sample 10

Input Files:
  Master data: Excel (.xlsx) or CSV with 'conversation_id' column
  ID filter:   CSV with conversation IDs in first column
  
Output:
  CSV with all rows from master data matching the conversation IDs
        """
    )
    
    parser.add_argument('master_file', help='Master data file (.xlsx or .csv)')
    parser.add_argument('id_file', help='CSV file with conversation IDs (first column)')
    parser.add_argument('output_file', help='Output CSV file')
    parser.add_argument('--save-missing', action='store_true',
                       help='Save conversation IDs not found to separate file')
    parser.add_argument('--show-sample', type=int, metavar='N',
                       help='Show first N rows of output as sample')
    
    args = parser.parse_args()
    
    try:
        print("=" * 80)
        print("CONVERSATION ID FILTER & EXTRACTION")
        print("=" * 80)
        
        # Load master data
        master_df = load_master_data(args.master_file)
        
        # Load conversation IDs to filter
        conversation_ids, id_column_name = load_conversation_ids(args.id_file)
        
        # Extract matching rows
        filtered_df, not_found_count, not_found_ids = extract_matching_rows(
            master_df, 
            conversation_ids
        )
        
        # Check if any rows were extracted
        if len(filtered_df) == 0:
            print("\n⚠️  Warning: No matching rows found!")
            print("   Check that conversation IDs in both files match exactly")
            
            # Show sample IDs from both files for debugging
            print("\n🔍 Sample conversation IDs from filter file:")
            sample_filter = list(conversation_ids)[:5]
            for cid in sample_filter:
                print(f"   '{cid}'")
            
            print("\n🔍 Sample conversation IDs from master data:")
            sample_master = master_df['conversation_id'].head(5).tolist()
            for cid in sample_master:
                print(f"   '{cid}'")
            
            response = input("\nContinue and save empty output? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("❌ Cancelled")
                sys.exit(0)
        
        # Save output
        save_output(
            filtered_df, 
            args.output_file, 
            not_found_count, 
            not_found_ids,
            args.save_missing
        )
        
        # Show sample output
        if args.show_sample and len(filtered_df) > 0:
            print(f"\n📋 Sample Output (first {args.show_sample} rows):")
            print(filtered_df.head(args.show_sample).to_string())
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\n📄 Files:")
        print(f"   Master data:   {args.master_file}")
        print(f"   Filter IDs:    {args.id_file}")
        print(f"   Output:        {args.output_file}")
        
        print(f"\n📊 Results:")
        print(f"   Rows extracted:           {len(filtered_df):,}")
        print(f"   Conversation IDs matched: {len(filtered_df['conversation_id'].unique()):,}")
        print(f"   Conversation IDs missing: {not_found_count:,}")
        
        if not_found_count > 0:
            print(f"\n⚠️  {not_found_count:,} conversation ID(s) were not found in master data")
            if args.save_missing:
                missing_file = args.output_file.replace('.csv', '_missing_ids.csv')
                print(f"   Missing IDs saved to: {missing_file}")
            else:
                print(f"   Use --save-missing to export list of missing IDs")
        
        print("\n💡 Next steps:")
        print(f"   1. Open {args.output_file}")
        print(f"   2. Verify extracted data")
        if not_found_count > 0:
            print(f"   3. Review missing conversation IDs")
        
        print()
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
