import re

def validate_plan_text(plan_text: str, agents_allowed):
    """
    Validate a plan written in the #TaskN/#AgentN/#DependencyN/#ExpectedOutputN format.

    Returns:
        (is_valid: bool, errors: list[str])
    """
    TASK_RE   = re.compile(r"^#Task(\d+): (.+)$", re.M)
    AGENT_RE  = re.compile(r"^#Agent(\d+): (.+)$", re.M)
    DEP_RE    = re.compile(r"^#Dependency(\d+): (.+)$", re.M)
    OUT_RE    = re.compile(r"^#ExpectedOutput(\d+): (.+)$", re.M)
    DEP_TOKEN = re.compile(r"#S(\d+)")

    errors = []

    # ------------------------------------------------------------
    # Parsing phase (feeds later checks)
    # - Not an error by itself, but missing lines are detected by _check_seq()
    # ------------------------------------------------------------
    tks  = TASK_RE.findall(plan_text)
    ags  = AGENT_RE.findall(plan_text)
    deps = DEP_RE.findall(plan_text)
    outs = OUT_RE.findall(plan_text)

    def _check_seq(pairs, label):
        # ------------------------------------------------------------
        # Error types handled here:
        # - MISSING_AGENT_LINES (via "Agent lines missing")
        # - BAD_TASK_NUMBERS   (via "Task numbers must be 1..N in order; got ...")
        #
        # (More generally: also catches missing/misnumbered Task/Dependency/ExpectedOutput lines.)
        # ------------------------------------------------------------
        if not pairs:
            errors.append(f"{label} lines missing")
            return
        nums = [int(n) for n, _ in pairs]
        if nums != list(range(1, len(nums) + 1)):
            errors.append(f"{label} numbers must be 1..N in order; got {nums}")

    # ------------------------------------------------------------
    # Section numbering checks
    # Error types:
    # - MISSING_AGENT_LINES (Agent lines missing)
    # - BAD_TASK_NUMBERS (Task numbers must be 1..N in order; got ...)
    # ------------------------------------------------------------
    _check_seq(tks,  "Task")
    _check_seq(ags,  "Agent")
    _check_seq(deps, "Dependency")
    _check_seq(outs, "ExpectedOutput")

    # ------------------------------------------------------------
    # Count consistency check across sections
    # Error type:
    # - MISSING_AGENT_LINES (often also triggers this, because Agent count becomes 0)
    # ------------------------------------------------------------
    if len({len(tks), len(ags), len(deps), len(outs)}) != 1:
        errors.append("Counts of Task/Agent/Dependency/ExpectedOutput must match")

    # ------------------------------------------------------------
    # Dependency syntax + range + direction checks
    # Error types:
    # - DEP_BAD_FORMAT     (DependencyN must be 'None' or '#S1 #S2 ...'; got 'INVALID_FORMAT')
    # - DEP_OUT_OF_RANGE   (DependencyN out of range [...]; valid 1..total)
    # - DEP_FORWARD_REF    (DependencyN forward reference [...]; only past steps allowed)
    #
    # Note: DEP_OUT_OF_RANGE can *also* trigger a forward-reference error in the same case
    #       (e.g., '#S3' when total=2 produces both out-of-range and forward-reference).
    # ------------------------------------------------------------
    if tks and deps:
        total = len(tks)
        for n, dep in deps:
            n = int(n)
            dep = dep.strip()
            if dep == "None":
                continue

            nums = [int(x) for x in DEP_TOKEN.findall(dep)]
            if not nums:
                errors.append(
                    f"Dependency{n} must be 'None' or '#S1 #S2 ...'; got '{dep}'"
                )
                continue

            bad = [k for k in nums if k < 1 or k > total]
            if bad:
                errors.append(
                    f"Dependency{n} out of range {bad}; valid 1..{total}"
                )

            fwd = [k for k in nums if k >= n]
            if fwd:
                errors.append(
                    f"Dependency{n} forward reference {fwd}; only past steps allowed"
                )

    # ------------------------------------------------------------
    # Agent name whitelist check
    # Error type:
    # - UNKNOWN_AGENT (AgentN unknown 'Bad Agent'. Allowed: [...])
    # ------------------------------------------------------------
    valid = set(agents_allowed)
    for n, name in AGENT_RE.findall(plan_text):
        if name not in valid:
            errors.append(
                f"Agent{n} unknown '{name}'. Allowed: {sorted(valid)}"
            )

    return (len(errors) == 0, errors)



# --------------------------------------------------------------------
# Base valid plan (your original example)
# --------------------------------------------------------------------
TEST_FINAL_PLAN = """\
#Task1: List the available IoT sites to confirm the existence of the MAIN facility.
#Agent1: IoT Data Download
#Dependency1: None
#ExpectedOutput1: A list of available IoT sites, confirming if MAIN is among them.

#Task2: List the assets at the MAIN site.
#Agent2: IoT Data Download
#Dependency2: #S1
#ExpectedOutput2: A list of assets located at the MAIN facility.
"""

# 1) Missing Agent lines  -> "Agent lines missing" + count mismatch
TEST_MISSING_AGENT_LINES = """\
#Task1: Do something.
#Dependency1: None
#ExpectedOutput1: Output for task 1.
"""

# 2) Bad Task numbering  -> "Task numbers must be 1..N in order"
TEST_BAD_TASK_NUMBERS = """\
#Task1: First task.
#Task3: Third task (skips 2).
#Agent1: IoT Data Download
#Agent2: IoT Data Download
#Dependency1: None
#Dependency2: #S1
#ExpectedOutput1: Out 1
#ExpectedOutput2: Out 2
"""

# 3) Dependency bad format -> "Dependency2 must be 'None' or '#S1 #S2 ...'"
TEST_DEP_BAD_FORMAT = """\
#Task1: First task.
#Task2: Second task.
#Agent1: IoT Data Download
#Agent2: IoT Data Download
#Dependency1: None
#Dependency2: INVALID_FORMAT
#ExpectedOutput1: Out 1
#ExpectedOutput2: Out 2
"""

# 4) Dependency out of range -> "Dependency2 out of range ..."
TEST_DEP_OUT_OF_RANGE = """\
#Task1: First task.
#Task2: Second task.
#Agent1: IoT Data Download
#Agent2: IoT Data Download
#Dependency1: None
#Dependency2: #S3
#ExpectedOutput1: Out 1
#ExpectedOutput2: Out 2
"""

# 5) Dependency forward reference -> "Dependency2 forward reference ..."
TEST_DEP_FORWARD_REF = """\
#Task1: First task.
#Task2: Second task.
#Agent1: IoT Data Download
#Agent2: IoT Data Download
#Dependency1: None
#Dependency2: #S2
#ExpectedOutput1: Out 1
#ExpectedOutput2: Out 2
"""

# 6) Unknown agent name -> "Agent2 unknown 'Bad Agent' ..."
TEST_UNKNOWN_AGENT = """\
#Task1: First task.
#Task2: Second task.
#Agent1: IoT Data Download
#Agent2: Bad Agent
#Dependency1: None
#Dependency2: #S1
#ExpectedOutput1: Out 1
#ExpectedOutput2: Out 2
"""


def run_case(name: str, plan: str, agents_allowed):
    print("=" * 70)
    print(f"TEST CASE: {name}")
    print("-" * 70)
    is_valid, errors = validate_plan_text(plan, agents_allowed)
    print("Plan valid:", is_valid)
    if errors:
        print("Errors:")
        for e in errors:
            print(" -", e)
    else:
        print("No errors found.")
    print()  # blank line


def main():
    agents_allowed = ["IoT Data Download"]

    cases = [
        ("VALID_PLAN", TEST_FINAL_PLAN),
        ("MISSING_AGENT_LINES", TEST_MISSING_AGENT_LINES),
        ("BAD_TASK_NUMBERS", TEST_BAD_TASK_NUMBERS),
        ("DEP_BAD_FORMAT", TEST_DEP_BAD_FORMAT),
        ("DEP_OUT_OF_RANGE", TEST_DEP_OUT_OF_RANGE),
        ("DEP_FORWARD_REF", TEST_DEP_FORWARD_REF),
        ("UNKNOWN_AGENT", TEST_UNKNOWN_AGENT),
    ]

    for name, plan in cases:
        run_case(name, plan, agents_allowed)


if __name__ == "__main__":
    main()
