import ast
import sys
sys.path.append('./src')
from data_utils import load_data, convert_to_qa_pairs

def test_converted_data():
    """Test that the converted health data works with existing parsing code"""
    
    print("=== Testing converted health data with existing parsing code ===")
    
    # Test loading the converted health data
    try:
        health_df = load_data('./data/health_data_converted.csv')
        print(f"✓ Successfully loaded health data: {health_df.shape}")
        
        # Test converting to QA pairs
        health_qa_pairs = convert_to_qa_pairs(health_df)
        print(f"✓ Successfully converted to QA pairs: {len(health_qa_pairs)} items")
        
        # Show a sample QA pair
        if health_qa_pairs:
            sample = health_qa_pairs[0]
            print(f"\n--- Sample QA Pair ---")
            print(f"Question: {sample['question'][:100]}...")
            print(f"Choices: {sample['choices']}")
            print(f"Answer: {sample['answer']}")
            print(f"Choices type: {type(sample['choices'])}")
        
        # Test with original cleaned data for comparison
        print(f"\n=== Testing original cleaned data ===")
        cleaned_df = load_data('./data/cleaned_data.csv')
        print(f"✓ Successfully loaded cleaned data: {cleaned_df.shape}")
        
        cleaned_qa_pairs = convert_to_qa_pairs(cleaned_df)
        print(f"✓ Successfully converted to QA pairs: {len(cleaned_qa_pairs)} items")
        
        if cleaned_qa_pairs:
            sample = cleaned_qa_pairs[0]
            print(f"\n--- Sample QA Pair ---")
            print(f"Question: {sample['question'][:100]}...")
            print(f"Choices: {sample['choices']}")
            print(f"Answer: {sample['answer']}")
            print(f"Choices type: {type(sample['choices'])}")
        
        print(f"\n✓ Both datasets work with the same parsing code!")
        print(f"✓ Health data: {len(health_qa_pairs)} questions")
        print(f"✓ Cleaned data: {len(cleaned_qa_pairs)} questions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_converted_data()
