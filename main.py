import math
import random
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

random.seed(7)

SKILLS = {
    "python_basics": {"name": "Python Basics", "prereq": []},
    "data_structs": {"name": "Data Structures", "prereq": ["python_basics"]},
    "numpy": {"name": "NumPy", "prereq": ["python_basics"]},
    "pandas": {"name": "Pandas", "prereq": ["numpy"]},
    "ml_basics": {"name": "ML Basics", "prereq": ["pandas"]},
    "classification": {"name": "Classification", "prereq": ["ml_basics"]},
    "regression": {"name": "Regression", "prereq": ["ml_basics"]},
    "model_eval": {"name": "Model Evaluation", "prereq": ["ml_basics"]},
}

CONTENT = [
    {"id": "c1", "title": "Intro to Python syntax", "skills": ["python_basics"], "difficulty": 1, "type": "reading"},
    {"id": "c2", "title": "Python exercises: variables & loops", "skills": ["python_basics"], "difficulty": 2, "type": "practice"},
    {"id": "c3", "title": "Lists, dicts, sets deep dive", "skills": ["data_structs"], "difficulty": 2, "type": "reading"},
    {"id": "c4", "title": "Build a CLI todo app", "skills": ["python_basics", "data_structs"], "difficulty": 3, "type": "project"},
    {"id": "c5", "title": "NumPy arrays & broadcasting", "skills": ["numpy"], "difficulty": 2, "type": "reading"},
    {"id": "c6", "title": "Pandas Series & DataFrames", "skills": ["pandas"], "difficulty": 3, "type": "reading"},
    {"id": "c7", "title": "Clean a CSV dataset", "skills": ["pandas"], "difficulty": 3, "type": "practice"},
    {"id": "c8", "title": "Intro to ML: train/test, features, targets", "skills": ["ml_basics"], "difficulty": 3, "type": "reading"},
    {"id": "c9", "title": "Linear Regression in scikit-learn", "skills": ["regression"], "difficulty": 3, "type": "practice"},
    {"id": "c10","title": "Logistic Regression & metrics", "skills": ["classification", "model_eval"], "difficulty": 4, "type": "reading"},
    {"id": "c11","title": "KNN classifier hands-on", "skills": ["classification"], "difficulty": 3, "type": "practice"},
    {"id": "c12","title": "Model evaluation: precision/recall/ROC", "skills": ["model_eval"], "difficulty": 4, "type": "reading"},
]

BKT_PARAMS = {
    "default": {"p_init": 0.2, "p_learn": 0.15, "p_guess": 0.2, "p_slip": 0.1},
    "model_eval": {"p_init": 0.15, "p_learn": 0.12, "p_guess": 0.15, "p_slip": 0.12},
}

def get_bkt_params(skill: str) -> Dict[str, float]:
    return BKT_PARAMS.get(skill, BKT_PARAMS["default"])

@dataclass
class UserModel:
    mastery: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.mastery:
            for s in SKILLS:
                self.mastery[s] = get_bkt_params(s)["p_init"]

    def prereq_ready(self, skill: str) -> float:
        prereqs = SKILLS[skill]["prereq"]
        if not prereqs:
            return 1.0
        vals = [self.mastery.get(p, 0.0) for p in prereqs]
        if not vals:
            return 1.0
        return max(0.0, min(vals))

    def update_with_response(self, item_skills: List[str], correct: bool):
        for s in item_skills:
            p = self.mastery.get(s, get_bkt_params(s)["p_init"])
            bp = get_bkt_params(s)
            p_guess, p_slip, p_learn = bp["p_guess"], bp["p_slip"], bp["p_learn"]
            if correct:
                numer = p * (1 - p_slip)
                denom = numer + (1 - p) * p_guess
            else:
                numer = p * p_slip
                denom = numer + (1 - p) * (1 - p_guess)
            if denom == 0:
                post = p
            else:
                post = numer / denom
            post = post + (1 - post) * p_learn
            self.mastery[s] = min(1.0, max(0.0, post))

EPSILON = 0.15

def expected_gain(user: UserModel, item: dict) -> float:
    gain = 0.0
    for s in item["skills"]:
        cur = user.mastery.get(s, 0.0)
        bp = get_bkt_params(s)
        readiness = user.prereq_ready(s)
        difficulty = item.get("difficulty", 3)
        diff_weight = math.exp(-((difficulty - 3.0) ** 2) / 3.0)
        gain += (1 - cur) * bp["p_learn"] * readiness * diff_weight
    return gain

def select_next_item(user: UserModel, seen: set) -> Optional[dict]:
    candidates = [it for it in CONTENT if it["id"] not in seen]
    if not candidates:
        return None
    if random.random() < EPSILON:
        candidates_sorted = sorted(
            candidates,
            key=lambda it: min(user.prereq_ready(s) for s in it["skills"]),
            reverse=True,
        )
        return candidates_sorted[0]
    scored = []
    for it in candidates:
        readiness_all = min(user.prereq_ready(s) for s in it["skills"])
        score = expected_gain(user, it) * readiness_all
        scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

TRUE_ABILITY = {s: random.uniform(0.3, 0.8) for s in SKILLS}

def simulate_correct(user: UserModel, item: dict) -> bool:
    probs = []
    for s in item["skills"]:
        base = 0.25 + 0.5 * user.mastery.get(s, 0.0) + 0.25 * TRUE_ABILITY[s]
        base *= 1 - 0.05 * (item.get("difficulty", 3) - 3)
        probs.append(max(0.05, min(0.95, base)))
    p_correct = sum(probs) / len(probs)
    return random.random() < p_correct

def run_path(num_steps: int = 10, simulate: bool = True) -> (UserModel, List[dict]):
    user = UserModel()
    seen = set()
    path = []
    for step in range(num_steps):
        nxt = select_next_item(user, seen)
        if not nxt:
            break
        seen.add(nxt["id"])
        if simulate:
            correct = simulate_correct(user, nxt)
        else:
            correct = False
        user.update_with_response(nxt["skills"], correct)
        path.append({
            "step": step + 1,
            "item_id": nxt["id"],
            "title": nxt["title"],
            "skills": nxt["skills"],
            "type": nxt["type"],
            "difficulty": nxt["difficulty"],
            "result": "correct" if correct else "incorrect",
            "mastery_snapshot": {s: round(user.mastery[s], 3) for s in nxt["skills"]},
        })
    return user, path
def main():
    print("Adaptive learning demo (simulated learner)")
    user, path = run_path(num_steps=12, simulate=True)
    with open("adaptive_learning_path.json", "w") as f:
        json.dump(path, f, indent=2)
    with open("skill_mastery_after.json", "w") as f:
        json.dump({s: round(user.mastery[s], 3) for s in SKILLS}, f, indent=2)
    print("Saved: adaptive_learning_path.json, skill_mastery_after.json")
    print("\nPath preview:")
    for p in path:
        print(f" {p['step']:02d}. {p['item_id']} - {p['title']} -> {p['result']} | mastery: {p['mastery_snapshot']}")

if __name__ == "__main__":
    main()