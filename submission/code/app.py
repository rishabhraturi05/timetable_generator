"""
Timetable Generator Web Application
-----------------------------------
A minimal, lightweight Flask app that generates clash-free academic timetables
using a Constraint Satisfaction Problem (CSP) Backtracking algorithm.

All constraints from the specification are satisfied:
1. Working days & Periods per day
2. Subject frequency (lectures/week)
3. Teacher availability & Teacher conflict prevention
4. Class conflict prevention
5. Classroom & Lab availability
6. Lab requirements (consecutive periods + specific room)
7. Fixed periods (e.g. Sports on Friday P5)
8. Teacher-subject mapping
9. Break periods (e.g. Lunch break)
10. Subject duration (consecutive periods, e.g. Workshop = 3 periods)
"""

import io
import csv
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

import re

def detect_group(name, explicit_grp=""):
    """Determines student sub-batch (e.g. A1, A2) or ALL for theory lectures."""
    if explicit_grp and str(explicit_grp).strip():
        val = str(explicit_grp).strip().upper()
        if val in ("ALL", "WHOLE", "CLASS", "-"):
            return "ALL"
        return val
    # Auto-detect pattern like (A1), (A2), B1, B2, Batch 1, Batch 2
    m = re.search(r'\b(A1|A2|B1|B2|BATCH\s*1|BATCH\s*2|GROUP\s*1|GROUP\s*2)\b', name, re.IGNORECASE)
    if m:
        return m.group(1).upper().replace(" ", "")
    m2 = re.search(r'\((A1|A2|B1|B2)\)', name, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return "ALL"

# ==============================================================================
# CORE ALGORITHM: Constraint Satisfaction Problem (CSP) Backtracking Solver
# ==============================================================================
def solve_timetable(config):
    """
    Takes a dictionary of constraints and generates clash-free schedules.
    Supports sub-batch parallel laboratory scheduling (e.g. A1 & A2 simultaneously).
    Returns:
        dict: {"success": True, "class_timetables": ..., "teacher_timetables": ...}
        or {"success": False, "error": "Description of why constraints cannot be satisfied"}
    """
    days = config.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    num_periods = int(config.get("periods_per_day", 6))
    breaks = set(int(b) for b in config.get("breaks", []))
    rooms = config.get("rooms", ["Room 101", "Room 102", "Physics Lab"])
    teachers = config.get("teachers", {})  # {teacher_name: ["Monday:1", ...]}
    classes = config.get("classes", [])
    subjects = config.get("subjects", [])

    # Periods available for teaching (excluding break periods)
    valid_periods = [p for p in range(1, num_periods + 1) if p not in breaks]
    total_slots_per_class = len(days) * len(valid_periods)

    # ---------------------------------------------------------
    # STEP 1: Pre-validation checks to detect impossible setups
    # ---------------------------------------------------------
    class_group_demands = {c: {} for c in classes}
    teacher_demands = {t: 0 for t in teachers}

    for sub in subjects:
        c = sub["class"]
        t = sub["teacher"]
        freq = int(sub.get("frequency", 1))
        dur = int(sub.get("duration", 1))
        grp = detect_group(sub.get("name", ""), sub.get("group", ""))
        total_needed = freq * dur

        if c not in class_group_demands:
            class_group_demands[c] = {}
        class_group_demands[c][grp] = class_group_demands[c].get(grp, 0) + total_needed
        teacher_demands[t] = teacher_demands.get(t, 0) + total_needed

    # Check: Did a class exceed total available slots (accounting for parallel sub-batches)?
    for c, grp_map in class_group_demands.items():
        all_demand = grp_map.get("ALL", 0)
        sub_max = max([v for k, v in grp_map.items() if k != "ALL"], default=0)
        effective_demand = all_demand + sub_max
        if effective_demand > total_slots_per_class:
            return {
                "success": False,
                "error": f"Class '{c}' requires {effective_demand} effective periods, but only {total_slots_per_class} teaching slots exist in the week."
            }

    # Check: Did a teacher's assigned periods exceed their available slots?
    for t, demand in teacher_demands.items():
        unavail = set(teachers.get(t, []))
        valid_unavail = sum(1 for slot in unavail if int(slot.split(":")[1]) not in breaks) if unavail else 0
        max_teacher_capacity = total_slots_per_class - valid_unavail
        if demand > max_teacher_capacity:
            return {
                "success": False,
                "error": f"Teacher '{t}' is assigned {demand} periods, but only has {max_teacher_capacity} available slots."
            }

    # ---------------------------------------------------------
    # STEP 2: Expand subjects into individual schedulable sessions
    # ---------------------------------------------------------
    sessions = []
    session_id = 0
    for sub in subjects:
        freq = int(sub.get("frequency", 1))
        dur = int(sub.get("duration", 1))
        fixed = sub.get("fixed")
        room = sub.get("room", "any")
        grp = detect_group(sub.get("name", ""), sub.get("group", ""))

        # Validate fixed slot if specified
        if fixed:
            try:
                f_day, f_per = fixed.split(":")
                f_per = int(f_per)
                if f_day not in days or f_per not in valid_periods:
                    return {
                        "success": False,
                        "error": f"Fixed slot '{fixed}' for '{sub['name']}' is invalid or falls on a break period."
                    }
                if f_per + dur - 1 > num_periods or any((f_per + offset) in breaks for offset in range(dur)):
                    return {
                        "success": False,
                        "error": f"Fixed slot '{fixed}' with duration {dur} for '{sub['name']}' overlaps with a break or exceeds day periods."
                    }
            except Exception:
                return {
                    "success": False,
                    "error": f"Invalid fixed slot format '{fixed}' for '{sub['name']}'. Use 'Day:Period' like 'Friday:5'."
                }

        # Create session units based on required frequency
        for i in range(freq):
            sessions.append({
                "id": session_id,
                "name": sub["name"],
                "class": sub["class"],
                "teacher": sub["teacher"],
                "duration": dur,
                "room": room,
                "group": grp,
                "fixed": fixed if i == 0 else None
            })
            session_id += 1

    # Heuristic: Schedule most-constrained items first:
    # 1. Sessions with fixed slots
    # 2. Longer duration sessions (e.g. Labs with 2 periods, Workshop with 3 periods)
    sessions.sort(key=lambda s: (0 if s["fixed"] else 1, -s["duration"]))

    # ---------------------------------------------------------
    # STEP 3: Setup timetable state grids for conflict checking
    # ---------------------------------------------------------
    # class_grid stores list of active sessions per slot to allow parallel sub-batches (e.g. A1 and A2)
    class_grid = {c: {d: {p: [] for p in range(1, num_periods + 1)} for d in days} for c in classes}
    teacher_grid = {t: {d: {p: None for p in range(1, num_periods + 1)} for d in days} for t in teachers}
    room_grid = {r: {d: {p: None for p in range(1, num_periods + 1)} for d in days} for r in rooms}

    # Lock break periods across all classes
    for d in days:
        for b in breaks:
            for c in classes:
                class_grid[c][d][b] = [{"name": "BREAK", "is_break": True}]

    teacher_unavail = {t: set(teachers.get(t, [])) for t in teachers}

    # ---------------------------------------------------------
    # STEP 4: Backtracking Search Functions
    # ---------------------------------------------------------
    max_steps = 50000
    steps = [0]

    def can_place(sess, day, start_p, room_chosen):
        """Checks hard constraints for placing a session, supporting sub-batches."""
        dur = sess["duration"]
        if start_p + dur - 1 > num_periods:
            return False

        cls = sess["class"]
        tch = sess["teacher"]
        grp = sess.get("group", "ALL")

        for p in range(start_p, start_p + dur):
            # Constraint: No lectures during breaks
            if p in breaks:
                return False

            # Constraint: Class / Sub-batch conflict
            cur_slots = class_grid[cls][day][p]
            if cur_slots:
                # Breaks
                if any(s.get("is_break") for s in cur_slots):
                    return False
                # If whole class is required (ALL), period must be completely unoccupied
                if grp == "ALL":
                    return False
                # If placing a sub-batch (e.g. A1), cannot place if an ALL lecture or same sub-batch is already scheduled
                if any(s.get("group") == "ALL" or s.get("group") == grp for s in cur_slots):
                    return False

            # Constraint: Teacher availability
            if f"{day}:{p}" in teacher_unavail.get(tch, set()):
                return False

            # Constraint: Teacher conflict
            if teacher_grid.get(tch, {}).get(day, {}).get(p) is not None:
                return False

            # Constraint: Classroom / Lab availability
            if room_chosen and room_chosen in room_grid:
                if room_grid[room_chosen][day][p] is not None:
                    return False

        return True

    def place_session(sess, day, start_p, room_chosen):
        """Occupies periods for the class (supporting parallel groups), teacher, and room."""
        dur = sess["duration"]
        cls = sess["class"]
        tch = sess["teacher"]
        info = {
            "session_id": sess["id"],
            "name": sess["name"],
            "teacher": tch,
            "room": room_chosen,
            "class": cls,
            "duration": dur,
            "start_period": start_p,
            "group": sess.get("group", "ALL")
        }
        for p in range(start_p, start_p + dur):
            class_grid[cls][day][p].append(info)
            if tch in teacher_grid:
                teacher_grid[tch][day][p] = info
            if room_chosen and room_chosen in room_grid:
                room_grid[room_chosen][day][p] = info

    def unplace_session(sess, day, start_p, room_chosen):
        """Releases occupied periods during backtrack."""
        dur = sess["duration"]
        cls = sess["class"]
        tch = sess["teacher"]
        for p in range(start_p, start_p + dur):
            class_grid[cls][day][p] = [s for s in class_grid[cls][day][p] if s.get("session_id") != sess["id"]]
            if tch in teacher_grid:
                teacher_grid[tch][day][p] = None
            if room_chosen and room_chosen in room_grid:
                room_grid[room_chosen][day][p] = None

    def backtrack(idx):
        """Recursive CSP backtracking algorithm."""
        steps[0] += 1
        if steps[0] > max_steps:
            return False

        if idx == len(sessions):
            return True

        sess = sessions[idx]
        req_room = sess["room"]
        candidate_rooms = [req_room] if (req_room and req_room != "any") else (rooms if rooms else ["General Room"])

        if sess["fixed"]:
            fixed_day, fixed_p = sess["fixed"].split(":")
            days_to_try = [fixed_day]
            periods_to_try = [int(fixed_p)]
        else:
            cls = sess["class"]
            days_to_try = sorted(days, key=lambda d: sum(1 for p in range(1, num_periods + 1)
                                                         if class_grid[cls][d][p] and not any(s.get("is_break") for s in class_grid[cls][d][p])))
            periods_to_try = valid_periods

        for day in days_to_try:
            for p in periods_to_try:
                if p + sess["duration"] - 1 > num_periods:
                    continue

                for rm in candidate_rooms:
                    if can_place(sess, day, p, rm):
                        place_session(sess, day, p, rm)
                        if backtrack(idx + 1):
                            return True
                        unplace_session(sess, day, p, rm)

        return False

    success = backtrack(0)

    if not success:
        return {
            "success": False,
            "error": "Cannot satisfy all constraints: A conflict occurred while attempting to schedule. "
                     "Check for overlapping teacher unavailabilities, tight room allocations, or excessive lecture counts."
        }

    # Format class_grid for frontend & CSV serialization
    formatted_class_grid = {}
    for c in classes:
        formatted_class_grid[c] = {}
        for d in days:
            formatted_class_grid[c][d] = {}
            for p in range(1, num_periods + 1):
                slots = class_grid[c][d][p]
                if not slots:
                    formatted_class_grid[c][d][p] = None
                elif len(slots) == 1:
                    formatted_class_grid[c][d][p] = slots[0]
                else:
                    formatted_class_grid[c][d][p] = {
                        "name": " / ".join(s["name"] for s in slots),
                        "teacher": " / ".join(s["teacher"] for s in slots),
                        "room": " / ".join(s["room"] for s in slots),
                        "class": c,
                        "is_parallel": True,
                        "parallel_sessions": slots
                    }

    return {
        "success": True,
        "class_timetables": formatted_class_grid,
        "teacher_timetables": teacher_grid,
        "days": days,
        "periods": list(range(1, num_periods + 1)),
        "breaks": list(breaks)
    }

# ==============================================================================
# FLASK WEB ROUTES
# ==============================================================================
@app.route("/")
def index():
    """Renders the main timetable website."""
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    """API endpoint to receive constraints JSON and return timetable result."""
    data = request.get_json() or {}
    result = solve_timetable(data)
    return jsonify(result)

@app.route("/api/download", methods=["POST"])
def api_download():
    """Generates a downloadable CSV of the timetable."""
    data = request.get_json() or {}
    result = solve_timetable(data)
    if not result.get("success"):
        return jsonify({"error": result.get("error")}), 400

    output = io.StringIO()
    writer = csv.writer(output)

    days = result["days"]
    periods = result["periods"]
    breaks = set(result["breaks"])
    timetables = result["class_timetables"]

    for class_name, days_data in timetables.items():
        writer.writerow([f"=== TIMETABLE FOR CLASS: {class_name} ==="])
        header = ["Day"] + [f"Period {p} (BREAK)" if p in breaks else f"Period {p}" for p in periods]
        writer.writerow(header)

        for d in days:
            row = [d]
            for p in periods:
                cell = days_data.get(d, {}).get(p)
                if not cell:
                    row.append("Free")
                elif cell.get("is_break"):
                    row.append("BREAK")
                elif cell.get("is_parallel") and cell.get("parallel_sessions"):
                    parts = [f"{s['name']} ({s['teacher']}, {s['room']})" for s in cell["parallel_sessions"]]
                    row.append(" // ".join(parts))
                else:
                    row.append(f"{cell['name']} ({cell['teacher']}, {cell['room']})")
            writer.writerow(row)
        writer.writerow([])  # blank separator line

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=timetable.csv"}
    )

if __name__ == "__main__":
    print("Starting Timetable Generator server on http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
