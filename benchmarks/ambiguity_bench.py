"""
Ambiguity Benchmark — EXP-001.

Measures whether QRR's collapse index χ discriminates between
ambiguous and unambiguous inputs without fine-tuning.

Hypothesis: Δχ = Mean χ(ambiguous) − Mean χ(clear) > 0.15
Success criterion: Δχ > 0.15, p < 0.05 (Mann-Whitney U)

Dataset:
  Ambiguous: lexical ambiguity, syntactic attachment ambiguity,
             referential ambiguity, pragmatic ambiguity.
  Clear:     unambiguous factual statements.

Usage:
  python benchmarks/ambiguity_bench.py --model gpt2 --branches 4
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from scipy import stats
from qrr.qrr_model import QRRModel

# ── Built-in sample dataset (100 + 100) ────────────────────────────────────

AMBIGUOUS_SAMPLES = [
    # Lexical ambiguity
    "The bank was steep and covered in moss.",
    "I went to the bank to deposit some checks.",
    "She saw the bat flying over the stadium.",
    "The pitcher was full of cold water.",
    "He read the book by the window.",  # read = past or present?
    "The professor said she would give the students a test on Friday.",
    "I need to get my glasses fixed.",
    "The chicken is ready to eat.",
    "The lamb is too young to eat.",
    "Visiting relatives can be boring.",
    # Syntactic attachment ambiguity
    "I saw the man with the telescope.",
    "The horse raced past the barn fell.",
    "The old man the boats.",
    "The complex houses married and single soldiers and their families.",
    "The cotton clothing is made of grows in Mississippi.",
    "She told the woman that she had met the man.",
    "The boy saw the girl with the binoculars.",
    "They are cooking apples.",
    "Time flies like an arrow.",
    "Fruit flies like a banana.",
    # Referential ambiguity
    "John told Bill that he should leave early.",
    "The dog bit the man because it was hungry.",
    "Mary saw Jane at the party. She was wearing a red dress.",
    "The trophy didn't fit in the suitcase because it was too big.",
    "The trophy didn't fit in the suitcase because it was too small.",
    "The police shot the rioters with guns.",
    "The developer argued with the designer because she thought the design was bad.",
    "After John ate the pizza, he threw the box in the trash.",
    "The secretary told the president that she would resign.",
    "The boy told the teacher that he had forgotten to do his homework.",
    # Pragmatic / scope ambiguity
    "Every student read some book.",
    "A nurse attended every patient.",
    "Someone loves everyone.",
    "The doctor said the patient will recover tomorrow.",
    "I didn't say he stole the money.",  # stress ambiguity
    "Only John passed the exam.",
    "Do you have a pen?",  # literal vs. speech act
    "Can you open the window?",
    "It's cold in here.",
    "He almost fell off the ladder.",
    # Named entity / word sense
    "The Mercury is in retrograde this month.",  # planet or element
    "Apple is considering a major acquisition.",
    "I need to take the bus to the terminal.",
    "She filed a complaint with the board.",
    "The plant was shut down by the inspector.",  # factory or organism
    "He addressed the letter to the president.",
    "The trunk was too heavy to carry.",  # car trunk, elephant trunk, luggage
    "She broke the record last Tuesday.",
    "The scale needs to be recalibrated.",
    "The kids are playing by the spring.",  # season or water spring
    # More complex
    "Book the flight or the hotel first.",
    "We discussed the problems with the house.",
    "The teacher called the student a genius.",
    "Susan told Mary that she had failed the exam.",
    "I will meet you at the bank at noon.",
    "The professor forgot to give the students their tests back.",
    "He saw her duck.",
    "She can't bear children.",
    "He found her attractive features.",
    "The captain addressed the crew.",
    # AmbigQA-style
    "When did Brazil last win the World Cup?",
    "Who wrote the national anthem?",
    "What is the meaning of life?",
    "How long did the war last?",
    "What is the capital of Georgia?",  # US state or country
    "When was the last election held?",
    "Who invented the telephone?",  # Bell vs Gray controversy
    "What is the population of China?",
    "How tall is the tower?",
    "When does summer start?",  # hemisphere ambiguity
    # Instruction ambiguity
    "Call me a taxi.",  # call FOR a taxi vs. call ME (name me) a taxi
    "Don't stop making noise.",
    "Let's eat, grandma.",
    "I saw the elephant in my pajamas.",
    "The chicken is ready to eat.",
    "He told me not to lie.",  # lie down or not tell lies
    "She can't help talking to herself.",
    "The duck is ready to eat.",
    "I'm going to the store.",  # grocery or retail?
    "The astronaut discovered a new planet.",  # which astronaut?
    # Multi-sense
    "He took the stand.",
    "They buried the hatchet.",
    "She hit the nail on the head.",
    "The company folded.",
    "He broke down crying.",
    "The bill was passed.",  # legislation or invoice
    "She saw the light.",  # literal or figurative
    "He's a real cold fish.",
    "She made her bed.",
    "He kicked the bucket.",
]

CLEAR_SAMPLES = [
    # Unambiguous facts
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth orbits the Sun once every 365.25 days.",
    "Paris is the capital of France.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Humans have 46 chromosomes.",
    "The Great Wall of China is over 21,000 kilometers long.",
    "Mount Everest is the highest mountain above sea level.",
    "The chemical formula for water is H₂O.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "The Eiffel Tower is located in Paris, France.",
    "The Amazon River is the largest river by discharge volume.",
    "Gold has atomic number 79.",
    "The human brain has approximately 86 billion neurons.",
    "The moon is approximately 384,400 kilometers from Earth.",
    "Oxygen is required for combustion to occur.",
    "The first moon landing occurred in 1969.",
    "DNA stands for deoxyribonucleic acid.",
    "The Pythagorean theorem states that a² + b² = c².",
    "Carbon dioxide is a greenhouse gas.",
    "The mitochondria is the powerhouse of the cell.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The boiling point of water increases with altitude.",
    "Gravity causes objects to fall toward Earth.",
    "The Sun is a star at the center of the solar system.",
    "Dogs are domesticated descendants of wolves.",
    "The human heart has four chambers.",
    "Antarctica is the coldest continent on Earth.",
    "Hydrogen is the lightest element.",
    "The speed of sound in air is approximately 343 meters per second.",
    "The Roman Empire fell in 476 AD.",
    "The Berlin Wall fell in 1989.",
    "Neil Armstrong was the first human to walk on the Moon.",
    "William Shakespeare was an English playwright.",
    "The International Space Station orbits Earth at about 400 kilometers altitude.",
    "Lions are the only social members of the cat family.",
    "The human body contains about 37 trillion cells.",
    "Antibiotics treat bacterial infections, not viral ones.",
    "The Earth's core is composed mainly of iron and nickel.",
    "Diamonds are made of carbon atoms.",
    "The Nile is the longest river in Africa.",
    "Python is a high-level programming language.",
    "The decimal number 10 equals binary 1010.",
    "TCP/IP is the fundamental communication protocol of the Internet.",
    "Madrid is the capital of Spain.",
    "Tokyo is the most populous metropolitan area in the world.",
    "The speed of light is faster than the speed of sound.",
    "Vaccines work by training the immune system.",
    "The Great Barrier Reef is located off the coast of Australia.",
    "Saturn has more moons than any other planet in the solar system.",
    "A triangle has three sides and three angles.",
    "Ultraviolet radiation can cause sunburn.",
    "The Internet was invented in the late 20th century.",
    "Albert Einstein developed the theory of relativity.",
    "Isaac Newton formulated the laws of motion and gravitation.",
    "The human skeleton has 206 bones in adults.",
    "Chlorophyll gives plants their green color.",
    "A leap year occurs every four years.",
    "The Atlantic Ocean separates North America from Europe.",
    "Binary stars are two stars that orbit a common center of mass.",
    "Rome was not built in a single year.",
    "The Louvre is a museum located in Paris.",
    "Elephants are the largest land animals.",
    "Penguins are native to the Southern Hemisphere.",
    "The human appendix is a vestigial organ.",
    "Light travels faster in a vacuum than in water.",
    "Electrons have negative charge.",
    "The periodic table was organized by Dmitri Mendeleev.",
    "A calorie is a unit of energy.",
    "The sun rises in the east and sets in the west.",
    "Bees are essential pollinators for many plants.",
    "The Milky Way is a barred spiral galaxy.",
    "Earthquakes are caused by tectonic plate movement.",
    "The human eye can distinguish approximately 10 million colors.",
    "Coffee contains caffeine.",
    "Salt is composed of sodium and chloride ions.",
    "The liver is the largest internal organ in the human body.",
    "Rainbows form when light is refracted through water droplets.",
    "The International System of Units is abbreviated SI.",
    "Volcanoes form at tectonic plate boundaries or hotspots.",
    "Alan Turing is considered the father of theoretical computer science.",
    "Neurons transmit information via electrical and chemical signals.",
    "The Wright Brothers made the first powered airplane flight in 1903.",
    "Marie Curie was the first person to win two Nobel Prizes.",
    "Black holes have gravitational pull so strong that light cannot escape.",
    "The Amazon rainforest produces about 20% of the world's oxygen.",
    "Insulin regulates blood sugar levels in the body.",
    "The four seasons are spring, summer, autumn, and winter.",
    "Proteins are made up of amino acids.",
    "The French Revolution began in 1789.",
    "Charles Darwin proposed the theory of evolution by natural selection.",
    "The speed of Earth's rotation is approximately 1,670 km/h at the equator.",
    "The Hubble Space Telescope was launched in 1990.",
    "Coral reefs are among the most biodiverse ecosystems on Earth.",
    "The Mediterranean Sea is bordered by Europe, Africa, and Asia.",
    "A virus is not a living organism by the classical definition.",
    "The square root of 144 is 12.",
    "Platinum is denser than gold.",
    "Stephen Hawking studied black holes and cosmology.",
]


def load_ambiguity_dataset(
    n: int = 100,
    custom_path: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Load ambiguity benchmark dataset.
    Returns (ambiguous_samples, clear_samples), each of length n.
    """
    if custom_path:
        with open(custom_path) as f:
            data = json.load(f)
        ambiguous = [d["text"] for d in data if d["is_ambiguous"]][:n]
        clear = [d["text"] for d in data if not d["is_ambiguous"]][:n]
    else:
        ambiguous = AMBIGUOUS_SAMPLES[:n]
        clear = CLEAR_SAMPLES[:n]
    return ambiguous, clear


def run_benchmark(
    model: QRRModel,
    n: int = 100,
    custom_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run EXP-001 benchmark.
    Returns full results dict.
    """
    ambiguous, clear = load_ambiguity_dataset(n=n, custom_path=custom_path)

    model.eval()
    chi_ambiguous = []
    chi_clear = []
    div_ambiguous = []
    div_clear = []

    if verbose:
        print(f"Running χ discrimination benchmark on {n} + {n} samples...")

    with torch.no_grad():
        for i, prompt in enumerate(ambiguous):
            out = model.forward_with_branches(prompt)
            chi_ambiguous.append(out["chi"])
            div_ambiguous.append(out["branch_diversity"])
            if verbose and (i + 1) % 20 == 0:
                print(f"  Ambiguous {i+1}/{n} — χ={out['chi']:.3f}")

        for i, prompt in enumerate(clear):
            out = model.forward_with_branches(prompt)
            chi_clear.append(out["chi"])
            div_clear.append(out["branch_diversity"])
            if verbose and (i + 1) % 20 == 0:
                print(f"  Clear     {i+1}/{n} — χ={out['chi']:.3f}")

    chi_a = np.array(chi_ambiguous)
    chi_c = np.array(chi_clear)
    delta_chi = chi_a.mean() - chi_c.mean()

    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(chi_a, chi_c, alternative="greater")

    success = delta_chi > 0.15 and p_value < 0.05

    results = {
        "n_ambiguous": n,
        "n_clear": n,
        "chi_ambiguous_mean": float(chi_a.mean()),
        "chi_ambiguous_std": float(chi_a.std()),
        "chi_clear_mean": float(chi_c.mean()),
        "chi_clear_std": float(chi_c.std()),
        "delta_chi": float(delta_chi),
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "success_criterion_met": success,
        "success_criterion": "Δχ > 0.15 AND p < 0.05",
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"EXP-001 Results")
        print(f"{'='*50}")
        print(f"χ ambiguous:  {chi_a.mean():.4f} ± {chi_a.std():.4f}")
        print(f"χ clear:      {chi_c.mean():.4f} ± {chi_c.std():.4f}")
        print(f"Δχ:           {delta_chi:.4f}")
        print(f"p-value:      {p_value:.4f} (Mann-Whitney U)")
        print(f"Success:      {'✅ YES' if success else '❌ NO (revise architecture)'}")
        print(f"Criterion:    Δχ > 0.15 AND p < 0.05")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QRR EXP-001 ambiguity benchmark")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--chi_threshold", type=float, default=0.3)
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    print(f"Loading QRR model ({args.model}, K={args.branches})...")
    model = QRRModel(
        base_model_name=args.model,
        k_branches=args.branches,
        chi_threshold=args.chi_threshold,
        freeze_base=True,
    )

    results = run_benchmark(model, n=args.n)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
