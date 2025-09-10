import csv
import json
import ast

def examine_data_formats():
    """Examine the format of both datasets"""
    print("=== CLEANED DATA FORMAT ===")
    with open('./data/cleaned_data.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"Headers: {headers}")
        
        # Show first few rows
        for i, row in enumerate(reader):
            if i < 3:
                print(f"\nRow {i+1}:")
                for key, value in row.items():
                    print(f"  {key}: {value}")
                    if key == 'choices':
                        # Parse the choices to see the format
                        try:
                            choices_parsed = ast.literal_eval(value)
                            print(f"    Parsed choices: {choices_parsed}")
                            print(f"    Type: {type(choices_parsed)}")
                        except:
                            print(f"    Could not parse choices")
            else:
                break
    
    print("\n\n=== HEALTH DATA FORMAT ===")
    with open('./data/health_data.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"Headers: {headers}")
        
        # Show first few rows
        for i, row in enumerate(reader):
            if i < 3:
                print(f"\nRow {i+1}:")
                for key, value in row.items():
                    print(f"  {key}: {value}")
                    if key == 'options':
                        # Parse the options to see the format
                        try:
                            options_parsed = json.loads(value)
                            print(f"    Parsed options: {options_parsed}")
                            print(f"    Type: {type(options_parsed)}")
                        except:
                            print(f"    Could not parse options")
            else:
                break

def convert_health_data_format():
    """Convert health_data.csv to match cleaned_data.csv format"""
    print("\n=== CONVERTING HEALTH DATA ===")
    
    converted_rows = []
    
    with open('./data/health_data.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            try:
                # Parse the options JSON
                options_dict = json.loads(row['options'])
                
                # Convert to list format like cleaned_data
                choices_list = []
                for key in sorted(options_dict.keys()):  # Sort to ensure consistent order (A, B, C, D)
                    choices_list.append(f"{key}. {options_dict[key]}")
                
                # Create new row in cleaned_data format
                new_row = {
                    'question': row['question'],
                    'choices': str(choices_list),  # Convert to string representation like cleaned_data
                    'answer': f"{row['answer_idx']}. {row['answer']}",  # Combine answer_idx and answer
                    'level': '',  # Leave blank as requested
                    'subject': ''  # Leave blank as requested
                }
                
                converted_rows.append(new_row)
                
                # Show first few conversions
                if i < 3:
                    print(f"\nConversion {i+1}:")
                    print(f"  Original options: {row['options']}")
                    print(f"  Converted choices: {new_row['choices']}")
                    print(f"  Original answer: {row['answer']} (idx: {row['answer_idx']})")
                    print(f"  Converted answer: {new_row['answer']}")
                
            except Exception as e:
                print(f"Error processing row {i+1}: {e}")
                continue
    
    # Save the converted data
    output_file = './data/health_data_converted.csv'
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['question', 'choices', 'answer', 'level', 'subject']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(converted_rows)
    
    print(f"\nConverted {len(converted_rows)} rows and saved to {output_file}")
    
    return output_file

if __name__ == "__main__":
    examine_data_formats()
    converted_file = convert_health_data_format()
