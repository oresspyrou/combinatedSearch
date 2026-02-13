import os
import subprocess
import sys

# Διαδρομές αρχείων
TREC_EVAL_EXE = ".\\trec_eval.exe"
QRELS_PATH = "data\\IR2025\\qrels.txt"
RESULTS_PATH = "data\\results\\hybrid_results.txt"

# Προσωρινά αρχεία (καθαρά)
TEMP_QRELS = "data\\temp_qrels_clean.txt"
TEMP_RESULTS = "data\\temp_results_clean.txt"

def clean_file(input_path, output_path):
    """Διαβάζει ένα αρχείο, αφαιρεί BOM και διορθώνει τα κενά."""
    try:
        with open(input_path, 'r', encoding='utf-8-sig') as f_in:
            lines = f_in.readlines()
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in lines:
                # Αντικατάσταση Tabs με κενά και αφαίρεση περιττών whitespaces
                parts = line.strip().split()
                if not parts: continue # Παράβλεψη κενών γραμμών
                
                # Ανακατασκευή της γραμμής με απλά κενά
                clean_line = " ".join(parts) + "\n"
                f_out.write(clean_line)
        
        print(f"✅ Cleaned file saved to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error cleaning {input_path}: {e}")
        return False

def run_evaluation():
    # 1. Καθαρισμός των αρχείων
    if not clean_file(QRELS_PATH, TEMP_QRELS): return
    if not clean_file(RESULTS_PATH, TEMP_RESULTS): return

    # 2. Εκτέλεση trec_eval με τα καθαρά αρχεία
    command = [TREC_EVAL_EXE, TEMP_QRELS, TEMP_RESULTS]
    
    print(f"\n🚀 Running: {' '.join(command)}")
    print("-" * 40)
    
    try:
        # Εκτέλεση και καταγραφή του αποτελέσματος
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Έλεγχος αν έβγαλε αποτέλεσμα
        if result.stdout:
            print(result.stdout)
        else:
            print("⚠️ No output produced!")
            if result.stderr:
                print("Error Output:", result.stderr)
                
    except FileNotFoundError:
        print("❌ Error: trec_eval.exe not found in the root folder!")
    except Exception as e:
        print(f"❌ Execution Error: {e}")

    # 3. Διαγραφή προσωρινών αρχείων (προαιρετικά)
    # if os.path.exists(TEMP_QRELS): os.remove(TEMP_QRELS)
    # if os.path.exists(TEMP_RESULTS): os.remove(TEMP_RESULTS)

if __name__ == "__main__":
    run_evaluation()