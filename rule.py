# rule_checker.py

from typing import List, Tuple, Dict
import pandas as pd
import re

# =========================
# 1. Phase allowed steps
# =========================

# Defines which steps are permitted within each specific Phase
PHASE_ALLOWED: Dict[str, List[str]] = {
    "P1": ["S1", "S2", "S3", "S14", "S22", "S39", "S40"],
    "P2": ["S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S22", "S39", "S40"],
    "P3": ["S12", "S13", "S39"],
    "P4": ["S15", "S16", "S17", "S18", "S19", "S26", "S39", "S40"],
    "P5": ["S3", "S22", "S23", "S24", "S25", "S39", "S40"],
    "P6": ["S3", "S20", "S21", "S39", "S40"],
    "P7": ["S27", "S28"],
    "P8": ["S29", "S30", "S31", "S32", "S33", "S34", "S35", "S39", "S40"],
    "P9": ["S36", "S37", "S38"],
    "P10": ["S39", "S40", "S43"],
    "P11": ["S41", "S42", "S44", "S45"],
}

# Key steps that "must appear at least once" in each phase
REQUIRED_STEPS: Dict[str, List[str]] = {
    "P1":  ["S2", "S3"],
    "P2":  ["S4", "S5", "S6", "S7", "S8", "S11"],
    "P3":  ["S12", "S13"],
    "P4":  ["S15", "S16", "S17", "S18", "S19"],
    "P5":  ["S22", "S23", "S24", "S25"],
    "P6":  ["S20", "S21"],
    "P7":  ["S27", "S28"],
    "P8":  ["S29", "S30", "S31", "S32", "S33"],
    "P9":  ["S36", "S37"],
    "P10": ["S39"],   # At least one coagulation
    "P11": ["S42"],   # Must include trocar_removal
}


# =========================
# 2. Utility Functions: Indexing and Comparison
# =========================

def normalize_sequence(seq) -> List[str]:
    """Supports 's1,s2,s3' or ['s1','s2'], unifying into ['S1','S2','S3']"""
    if isinstance(seq, str):
        parts = [x.strip() for x in seq.split(",") if x.strip()]
    else:
        parts = [str(x).strip() for x in seq if str(x).strip()]
    return [p.upper() for p in parts]


def build_indices(seq: List[str]):
    """Returns dictionaries for first_idx and last_idx of steps in sequence"""
    first_idx = {}
    last_idx = {}
    for i, s in enumerate(seq):
        if s not in first_idx:
            first_idx[s] = i
        last_idx[s] = i
    return first_idx, last_idx


def require_first_before(
    first_idx: Dict[str, int],
    a: str,
    b: str,
    reasons: List[str],
    rule_name: str,
    cond: bool = True,
):
    """Checks if FIRST occurrence of 'a' < FIRST occurrence of 'b' (only if cond=True and both exist)"""
    if not cond:
        return
    if a in first_idx and b in first_idx:
        if not (first_idx[a] < first_idx[b]):
            reasons.append(f"{rule_name}: FIRST {a} must be before FIRST {b}.")


def require_after(
    first_idx: Dict[str, int],
    a: str,
    b: str,
    reasons: List[str],
    rule_name: str,
    cond: bool = True,
):
    """Checks if FIRST occurrence of 'a' > FIRST occurrence of 'b' ('a' must appear after 'b')"""
    if not cond:
        return
    if a in first_idx and b in first_idx:
        if not (first_idx[a] > first_idx[b]):
            reasons.append(f"{rule_name}: FIRST {a} must be after FIRST {b}.")


def forbid_after_last(
    seq: List[str],
    last_idx: Dict[str, int],
    anchor: str,
    forbidden: List[str],
    reasons: List[str],
    rule_name: str,
):
    """Checks: No step in the 'forbidden' list may appear after the LAST occurrence of 'anchor'"""
    if anchor not in last_idx:
        return
    last_pos = last_idx[anchor]
    for i in range(last_pos + 1, len(seq)):
        s = seq[i]
        if s in forbidden:
            reasons.append(
                f"{rule_name}: {s} must NOT appear after LAST {anchor} (found at position {i})."
            )


# =========================
# 3. Rule Checks per Phase
# =========================

def check_rules_p1(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # 1. If S1 appears: FIRST S1 < FIRST S2.
    require_first_before(first_idx, "S1", "S2", reasons, "P1-1")

    # 2. FIRST S2 < FIRST S3.
    require_first_before(first_idx, "S2", "S3", reasons, "P1-2", cond=("S3" in first_idx))

    # 3. If S14 appears: after S2 and S3.
    require_after(first_idx, "S14", "S2", reasons, "P1-3a")
    require_after(first_idx, "S14", "S3", reasons, "P1-3b")

    # 4. If S22 appears: after S2 and S3.
    require_after(first_idx, "S22", "S2", reasons, "P1-4a")
    require_after(first_idx, "S22", "S3", reasons, "P1-4b")

    # 5. S39/S40 after S3.
    require_after(first_idx, "S39", "S3", reasons, "P1-5a", cond=("S39" in first_idx))
    require_after(first_idx, "S40", "S3", reasons, "P1-5b", cond=("S40" in first_idx))


def check_rules_p2(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # 1. If S5 appears: FIRST S4 < FIRST S5.
    require_first_before(first_idx, "S4", "S5", reasons, "P2-1", cond=("S5" in first_idx))

    # 2. If S6 appears: FIRST S5 < FIRST S6.
    require_first_before(first_idx, "S5", "S6", reasons, "P2-2", cond=("S6" in first_idx))

    # 3. If S8 appears: FIRST S6 < FIRST S8.
    require_first_before(first_idx, "S6", "S8", reasons, "P2-3", cond=("S8" in first_idx))

    # 4. If S11 appears: FIRST S8 < FIRST S11.
    require_first_before(first_idx, "S8", "S11", reasons, "P2-4", cond=("S11" in first_idx))

    # 5. S9/S10 after S6.
    require_after(first_idx, "S9", "S6", reasons, "P2-5a", cond=("S9" in first_idx))
    require_after(first_idx, "S10", "S6", reasons, "P2-5b", cond=("S10" in first_idx))

    # 6. S22 after S11.
    require_after(first_idx, "S22", "S11", reasons, "P2-6", cond=("S22" in first_idx))

    # 7. S39/S40 after S4.
    require_after(first_idx, "S39", "S4", reasons, "P2-7a", cond=("S39" in first_idx))
    require_after(first_idx, "S40", "S4", reasons, "P2-7b", cond=("S40" in first_idx))

    # 8. After LAST S11: no S4,S5,S6,S7,S8,S9,S10.
    forbid_after_last(
        seq,
        last_idx,
        "S11",
        ["S4", "S5", "S6", "S7", "S8", "S9", "S10"],
        reasons,
        "P2-8",
    )

    # 9. After LAST S8: no S6.
    forbid_after_last(seq, last_idx, "S8", ["S6"], reasons, "P2-9")

    # 10. After LAST S8: no S7.
    forbid_after_last(seq, last_idx, "S8", ["S7"], reasons, "P2-10")


def check_rules_p3(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # 1. If S13 appears: FIRST S12 < FIRST S13.
    require_first_before(first_idx, "S12", "S13", reasons, "P3-1", cond=("S13" in first_idx))

    # 2. If S39 appears: FIRST S39 after FIRST S12.
    require_after(first_idx, "S39", "S12", reasons, "P3-2", cond=("S39" in first_idx))


def check_rules_p4(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # 1. FIRST S15 < FIRST S16 (if both).
    require_first_before(first_idx, "S15", "S16", reasons, "P4-1", cond=("S15" in first_idx and "S16" in first_idx))

    # 2. FIRST S16 < FIRST S17 (if both).
    require_first_before(first_idx, "S16", "S17", reasons, "P4-2", cond=("S16" in first_idx and "S17" in first_idx))

    # 3. FIRST S17 < FIRST S18 (if both).
    require_first_before(first_idx, "S17", "S18", reasons, "P4-3", cond=("S17" in first_idx and "S18" in first_idx))

    # 4. FIRST S18 < FIRST S19 (if both).
    require_first_before(first_idx, "S18", "S19", reasons, "P4-4", cond=("S18" in first_idx and "S19" in first_idx))

    # 5. S26 after S18.
    require_after(first_idx, "S26", "S18", reasons, "P4-5", cond=("S26" in first_idx))

    # 6. S39/S40 after S15.
    require_after(first_idx, "S39", "S15", reasons, "P4-6a", cond=("S39" in first_idx))
    require_after(first_idx, "S40", "S15", reasons, "P4-6b", cond=("S40" in first_idx))

    # 7. After LAST S19: no S17,S18.
    forbid_after_last(seq, last_idx, "S19", ["S17", "S18"], reasons, "P4-7")

    # 8. After LAST S26: no S18.
    forbid_after_last(seq, last_idx, "S26", ["S18"], reasons, "P4-8")

    # 9. After LAST S17: no S16.
    forbid_after_last(seq, last_idx, "S17", ["S16"], reasons, "P4-9")


def check_rules_p5(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # 1. FIRST S23 < FIRST S24 (if both).
    require_first_before(first_idx, "S23", "S24", reasons, "P5-1", cond=("S23" in first_idx and "S24" in first_idx))

    # 2. FIRST S24 < FIRST S25 (if both).
    require_first_before(first_idx, "S24", "S25", reasons, "P5-2", cond=("S24" in first_idx and "S25" in first_idx))

    # 3. FIRST S22 < FIRST S24 (if S24 exists).
    require_first_before(first_idx, "S22", "S24", reasons, "P5-3", cond=("S24" in first_idx and "S22" in first_idx))

    # 4. S39/S40 after S23.
    require_after(first_idx, "S39", "S23", reasons, "P5-4a", cond=("S39" in first_idx))
    require_after(first_idx, "S40", "S23", reasons, "P5-4b", cond=("S40" in first_idx))

    # 5. After LAST S25: no S22,S23,S24.
    forbid_after_last(seq, last_idx, "S25", ["S22", "S23", "S24"], reasons, "P5-5")


def check_rules_p6(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # 1. FIRST S20 < FIRST S21 (if both).
    require_first_before(first_idx, "S20", "S21", reasons, "P6-1", cond=("S20" in first_idx and "S21" in first_idx))

    # 2. S39/S40 after S20.
    require_after(first_idx, "S39", "S20", reasons, "P6-2a", cond=("S39" in first_idx))
    require_after(first_idx, "S40", "S20", reasons, "P6-2b", cond=("S40" in first_idx))

    # 3. After LAST S21: no S20.
    forbid_after_last(seq, last_idx, "S21", ["S20"], reasons, "P6-3")


def check_rules_p7(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # 1. FIRST S27 < FIRST S28 (if both).
    require_first_before(first_idx, "S27", "S28", reasons, "P7-1", cond=("S27" in first_idx and "S28" in first_idx))

    # 2. After LAST S28: no S27.
    forbid_after_last(seq, last_idx, "S28", ["S27"], reasons, "P7-2")


def check_rules_p8(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # 1. S29 < S30; S29 < S31.
    require_first_before(first_idx, "S29", "S30", reasons, "P8-1a", cond=("S29" in first_idx and "S30" in first_idx))
    require_first_before(first_idx, "S29", "S31", reasons, "P8-1b", cond=("S29" in first_idx and "S31" in first_idx))

    # 2. S29 < S32.
    require_first_before(first_idx, "S29", "S32", reasons, "P8-2", cond=("S29" in first_idx and "S32" in first_idx))

    # 3. S30 < S31.
    require_first_before(first_idx, "S30", "S31", reasons, "P8-3", cond=("S30" in first_idx and "S31" in first_idx))

    # 4. S31 < S32.
    require_first_before(first_idx, "S31", "S32", reasons, "P8-4", cond=("S31" in first_idx and "S32" in first_idx))

    # 5. S32 < S33.
    require_first_before(first_idx, "S32", "S33", reasons, "P8-5", cond=("S32" in first_idx and "S33" in first_idx))

    # 6. S34/S35 after S33.
    require_after(first_idx, "S34", "S33", reasons, "P8-6a", cond=("S34" in first_idx))
    require_after(first_idx, "S35", "S33", reasons, "P8-6b", cond=("S35" in first_idx))

    # 7. S39/S40 after S29.
    require_after(first_idx, "S39", "S29", reasons, "P8-7a", cond=("S39" in first_idx))
    require_after(first_idx, "S40", "S29", reasons, "P8-7b", cond=("S40" in first_idx))

    # 8. After LAST S33: no S29,S30,S31,S32.
    forbid_after_last(seq, last_idx, "S33", ["S29", "S30", "S31", "S32"], reasons, "P8-8")

    # 9. After LAST S32: no S29,S30,S31.
    forbid_after_last(seq, last_idx, "S32", ["S29", "S30", "S31"], reasons, "P8-9")


def check_rules_p9(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # 1. S36 < S37.
    require_first_before(first_idx, "S36", "S37", reasons, "P9-1", cond=("S36" in first_idx and "S37" in first_idx))

    # 2. S37 < S38.
    require_first_before(first_idx, "S37", "S38", reasons, "P9-2", cond=("S37" in first_idx and "S38" in first_idx))

    # 3. After LAST S37: no S36.
    forbid_after_last(seq, last_idx, "S37", ["S36"], reasons, "P9-3")


def check_rules_p10(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # No specific ordering rules for P10
    pass


def check_rules_p11(seq: List[str], first_idx, last_idx, reasons: List[str]):
    # 1. FIRST S41 < FIRST S42.
    require_first_before(first_idx, "S41", "S42", reasons, "P11-1", cond=("S41" in first_idx and "S42" in first_idx))

    # 2. After LAST S42: no S41,S44,S45.
    forbid_after_last(seq, last_idx, "S42", ["S41", "S44", "S45"], reasons, "P11-2")


PHASE_RULES = {
    "P1": check_rules_p1,
    "P2": check_rules_p2,
    "P3": check_rules_p3,
    "P4": check_rules_p4,
    "P5": check_rules_p5,
    "P6": check_rules_p6,
    "P7": check_rules_p7,
    "P8": check_rules_p8,
    "P9": check_rules_p9,
    "P10": check_rules_p10,
    "P11": check_rules_p11,
}


# =========================
# 4. Main Function: External Interface
# =========================

def check_sequence(phase: str, sequence) -> Tuple[bool, List[str]]:
    """
    Validates if a (phase, sequence) pair conforms to all rules.

    Parameters:
        phase: "P1"..."P11"
        sequence: "s1,s2,s3" or ["s1","s2","s3"]

    Returns:
        (is_valid, reasons)
        is_valid: True if all rules pass.
        reasons: If False, contains a list of rule violation descriptions.
    """
    phase = phase.strip().upper()
    if phase not in PHASE_ALLOWED:
        return False, [f"Unknown phase: {phase}"]

    seq = normalize_sequence(sequence)
    reasons: List[str] = []

    # 1) allowed-step check
    allowed = set(PHASE_ALLOWED[phase])
    for i, s in enumerate(seq):
        if s not in allowed:
            reasons.append(f"Allowed-set violation: step {s} at position {i} not allowed in {phase}.")

    # 2) Required-step check
    required = REQUIRED_STEPS.get(phase, [])
    present = set(seq)
    for r in required:
        if r not in present:
            reasons.append(
                f"Required-step violation: step {r} must appear at least once in {phase}."
            )

    # 3) Order-related checks
    first_idx, last_idx = build_indices(seq)
    rule_func = PHASE_RULES[phase]
    rule_func(seq, first_idx, last_idx, reasons)

    is_valid = len(reasons) == 0
    return is_valid, reasons


# =========================
# 5. Simple Test / Processing
# =========================

in_path  = r"D:\software\qq\results\clean_VLM\task1\clean_Lingshu-32B_task1.xlsx"
out_path = r"D:\software\qq\results\VLM_final\task1\merged_Lingshu-32B_task1clean.xlsx"

def normalize_cell(x):
    if x is None:
        return ""
    x = str(x).strip()
    if x.lower() in {"nan", "none"}:
        return ""
    return x


def split_steps(x):
    """
    Splits cell content into a step list.
    Supports commas / semicolons / newlines / list strings.
    """
    x = normalize_cell(x)
    if not x:
        return []

    x = x.strip("[](){}")
    parts = re.split(r"[,\n;]+", x)

    return [p.strip().strip("'\"") for p in parts if p.strip()]


def remove_s0(steps):
    """
    Removes steps starting with s0 (e.g., s0, S0, s0_xxx, s0 xxx)
    """
    clean = []
    for s in steps:
        if re.match(r"(?i)^s0($|[_\s])", s):
            continue
        clean.append(s)
    return clean


def build_sequence(*cells):
    """Combines multiple cells into a single comma-separated step sequence"""
    steps = []
    for c in cells:
        steps.extend(split_steps(c))
    steps = remove_s0(steps)
    return ", ".join(steps)


def to_yes_no_error_with_reason(phase, seq):
    """Helper for converting validation results to sheet labels"""
    if not phase or not seq:
        return "error", "empty phase or step sequence"

    is_valid, messages = check_sequence(phase, seq)

    label = "yes" if is_valid else "no"
    reason = "; ".join(messages) if messages else ""

    return label, reason


def process_file(input_path, output_path):
    """Processes the Excel file and saves the validation results"""
    df = pd.read_excel(input_path)
    df.columns = [c.strip() for c in df.columns]

    df["C_check"] = ""
    df["N_check"] = ""

    for idx, row in df.iterrows():
        phase = normalize_cell(row.get("phase", ""))

        # ---------- C Sequence Processing ----------
        c_seq = build_sequence(
            row.get("previous_steps_in_phase_clean", ""),
            row.get("VLM_C_Step", ""),
            row.get("Remaining_Steps", "")
        )
        df.at[idx, "C_sequences"] = c_seq

        c_label, c_reason = to_yes_no_error_with_reason(phase, c_seq)
        df.at[idx, "C_check"] = c_label

        # ---------- N Sequence Processing ----------
        n_phase = normalize_cell(row.get("VLM_N_Pahse", "")) # Note: "Pahse" typo preserved from original column name
        n_seq = build_sequence(row.get("VLM_N_Step", ""))
        df.at[idx, "N_sequences"] = n_seq

        n_label, n_reason = to_yes_no_error_with_reason(n_phase, n_seq)
        df.at[idx, "N_check"] = n_label

    df.to_excel(output_path, index=False)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    process_file(in_path, out_path)