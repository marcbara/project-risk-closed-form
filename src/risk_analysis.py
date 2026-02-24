"""
Numerical Validation: Cornish-Fisher Approximation vs Monte Carlo Simulation
for Project Risk Contingency Reserves

Author: Marc Bara
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple
import warnings
from pathlib import Path

# Project root is one level up from this script (src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Risk:
    """Represents a single project risk."""
    id: str
    name: str
    probability: float  # p_i
    q10: float          # 10th percentile of impact
    q90: float          # 90th percentile of impact
    is_threat: bool     # True for threat (+), False for opportunity (-)
    
    def __post_init__(self):
        # Compute log-normal parameters
        self.mu = 0.5 * (np.log(self.q90) + np.log(self.q10))
        self.sigma = (np.log(self.q90) - np.log(self.q10)) / (2 * stats.norm.ppf(0.90))
        self.sign = 1 if self.is_threat else -1
        
    @property
    def lognormal_mean(self) -> float:
        """E[D_i] = exp(mu + sigma^2/2)"""
        return np.exp(self.mu + self.sigma**2 / 2)
    
    @property
    def lognormal_variance(self) -> float:
        """Var[D_i]"""
        return (np.exp(self.sigma**2) - 1) * np.exp(2*self.mu + self.sigma**2)


# =============================================================================
# Moment Calculations
# =============================================================================

def compute_individual_moments(risk: Risk) -> dict:
    """
    Compute the first four moments of X_i = B_i * D_i * S_i
    
    Returns dict with: mean, variance, nu3 (3rd central moment), nu4 (4th central moment)
    """
    p = risk.probability
    mu = risk.mu
    sigma = risk.sigma
    S = risk.sign
    
    # Raw moments of log-normal: E[D^k] = exp(k*mu + k^2*sigma^2/2)
    M1 = np.exp(mu + sigma**2 / 2)
    M2 = np.exp(2*mu + 2*sigma**2)
    M3 = np.exp(3*mu + 9*sigma**2 / 2)
    M4 = np.exp(4*mu + 8*sigma**2)
    
    # Mean of X_i
    mean = S * p * M1
    
    # E[X_i^2] = p * M2 (since S^2 = 1)
    EX2 = p * M2
    
    # Variance
    variance = EX2 - mean**2
    
    # E[X_i^3] = S * p * M3
    EX3 = S * p * M3
    
    # E[X_i^4] = p * M4
    EX4 = p * M4
    
    # Third central moment: E[(X-m)^3] = E[X^3] - 3*m*E[X^2] + 2*m^3
    nu3 = EX3 - 3*mean*EX2 + 2*mean**3
    
    # Fourth central moment: E[(X-m)^4] = E[X^4] - 4*m*E[X^3] + 6*m^2*E[X^2] - 3*m^4
    nu4 = EX4 - 4*mean*EX3 + 6*mean**2*EX2 - 3*mean**4
    
    return {
        'mean': mean,
        'variance': variance,
        'nu3': nu3,
        'nu4': nu4,
        'M1': M1,
        'M2': M2,
        'M3': M3,
        'M4': M4
    }


def compute_aggregate_moments(risks: List[Risk]) -> dict:
    """
    Compute aggregate moments for sum of independent risks.
    
    For independent variables, cumulants add:
    - κ₁(X) = Σ κ₁(Xᵢ) = Σ μᵢ  (mean)
    - κ₂(X) = Σ κ₂(Xᵢ) = Σ σᵢ² (variance)
    - κ₃(X) = Σ κ₃(Xᵢ) = Σ ν₃ᵢ (third cumulant = third central moment)
    - κ₄(X) = Σ κ₄(Xᵢ) where κ₄ᵢ = ν₄ᵢ - 3σᵢ⁴ (fourth cumulant)
    
    Then: ν₄(X) = κ₄(X) + 3[κ₂(X)]² = Σκ₄ᵢ + 3(Σσᵢ²)²
    
    Returns dict with: mean, variance, std, skewness (gamma1), excess_kurtosis (gamma2)
    """
    total_mean = 0.0
    total_variance = 0.0
    total_kappa3 = 0.0  # Third cumulant
    total_kappa4 = 0.0  # Fourth cumulant
    
    for risk in risks:
        moments = compute_individual_moments(risk)
        total_mean += moments['mean']
        total_variance += moments['variance']
        total_kappa3 += moments['nu3']  # κ₃ = ν₃ (third cumulant equals third central moment)
        # Fourth cumulant: κ₄ = ν₄ - 3σ⁴
        kappa4_i = moments['nu4'] - 3 * moments['variance']**2
        total_kappa4 += kappa4_i
    
    std = np.sqrt(total_variance)
    
    # Standardized moments
    gamma1 = total_kappa3 / (std**3) if std > 0 else 0  # Skewness = κ₃/σ³
    gamma2 = total_kappa4 / (std**4) if std > 0 else 0  # Excess kurtosis = κ₄/σ⁴
    
    # Also compute total ν₄ for reference
    total_nu4 = total_kappa4 + 3 * total_variance**2
    
    return {
        'mean': total_mean,
        'variance': total_variance,
        'std': std,
        'gamma1': gamma1,
        'gamma2': gamma2,
        'kappa3': total_kappa3,
        'kappa4': total_kappa4,
        'nu4': total_nu4
    }


# =============================================================================
# Cornish-Fisher Approximation
# =============================================================================

def cornish_fisher_z(alpha: float, gamma1: float, gamma2: float) -> float:
    """
    Compute the Cornish-Fisher adjusted z-value.
    
    z_CF = z + (z^2-1)/6 * gamma1 + (z^3-3z)/24 * gamma2 - (2z^3-5z)/36 * gamma1^2
    """
    z = stats.norm.ppf(alpha)
    
    z_cf = (z 
            + (z**2 - 1) / 6 * gamma1 
            + (z**3 - 3*z) / 24 * gamma2 
            - (2*z**3 - 5*z) / 36 * gamma1**2)
    
    return z_cf


def cornish_fisher_quantile(alpha: float, mean: float, std: float, 
                            gamma1: float, gamma2: float) -> float:
    """
    Compute the alpha-quantile using Cornish-Fisher approximation.
    """
    z_cf = cornish_fisher_z(alpha, gamma1, gamma2)
    return mean + std * z_cf


def check_cf_validity(gamma1: float, gamma2: float) -> Tuple[bool, str]:
    """
    Check if Cornish-Fisher expansion is valid for given skewness/kurtosis.
    """
    warnings_list = []
    valid = True
    
    # Maillard (2012) bounds
    if abs(gamma1) > 2.49:
        valid = False
        warnings_list.append(f"Skewness |{gamma1:.2f}| > 2.49 exceeds validity bound")
    
    if gamma2 > 6:
        warnings_list.append(f"Excess kurtosis {gamma2:.2f} > 6 may cause issues")
    
    if abs(gamma1) > 1.5:
        warnings_list.append(f"Skewness |{gamma1:.2f}| > 1.5: accuracy may degrade")
    
    return valid, "; ".join(warnings_list) if warnings_list else "OK"


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

def monte_carlo_simulation(risks: List[Risk], n_simulations: int = 1_000_000,
                           seed: int = 42) -> np.ndarray:
    """
    Run Monte Carlo simulation for aggregate risk.
    
    Returns array of simulated total impacts.
    """
    np.random.seed(seed)
    n_risks = len(risks)
    
    # Generate all Bernoulli outcomes at once
    bernoulli_matrix = np.random.random((n_simulations, n_risks))
    
    # Generate all log-normal impacts at once
    total_impact = np.zeros(n_simulations)
    
    for i, risk in enumerate(risks):
        # Bernoulli: 1 if random < p, else 0
        occurs = (bernoulli_matrix[:, i] < risk.probability).astype(float)
        
        # Log-normal impact (only matters when risk occurs)
        impact = np.random.lognormal(risk.mu, risk.sigma, n_simulations)
        
        # Add to total (with sign)
        total_impact += occurs * impact * risk.sign
    
    return total_impact


def compute_mc_statistics(simulations: np.ndarray, alpha: float = 0.90) -> dict:
    """
    Compute statistics from Monte Carlo simulations.
    """
    return {
        'mean': np.mean(simulations),
        'std': np.std(simulations),
        'skewness': stats.skew(simulations),
        'kurtosis': stats.kurtosis(simulations),  # Excess kurtosis
        'quantile': np.percentile(simulations, alpha * 100),
        'min': np.min(simulations),
        'max': np.max(simulations),
        'p10': np.percentile(simulations, 10),
        'p50': np.percentile(simulations, 50),
        'p75': np.percentile(simulations, 75),
        'p90': np.percentile(simulations, 90),
        'p95': np.percentile(simulations, 95),
        'p99': np.percentile(simulations, 99)
    }


# =============================================================================
# Comparison Function
# =============================================================================

def compare_methods(risks: List[Risk], alpha: float = 0.90,
                   n_simulations: int = 1_000_000) -> dict:
    """
    Compare Cornish-Fisher approximation with Monte Carlo simulation.
    """
    # Analytical moments
    moments = compute_aggregate_moments(risks)
    
    # Cornish-Fisher quantile
    cf_quantile = cornish_fisher_quantile(
        alpha, moments['mean'], moments['std'], 
        moments['gamma1'], moments['gamma2']
    )
    
    # Check validity
    cf_valid, cf_warning = check_cf_validity(moments['gamma1'], moments['gamma2'])
    
    # Monte Carlo
    mc_simulations = monte_carlo_simulation(risks, n_simulations)
    mc_stats = compute_mc_statistics(mc_simulations, alpha)
    
    # Relative error
    mc_quantile = mc_stats['quantile']
    if mc_quantile != 0:
        rel_error = (cf_quantile - mc_quantile) / abs(mc_quantile)
    else:
        rel_error = np.nan
    
    return {
        'n_risks': len(risks),
        'n_threats': sum(1 for r in risks if r.is_threat),
        'n_opportunities': sum(1 for r in risks if not r.is_threat),
        'analytical_mean': moments['mean'],
        'analytical_std': moments['std'],
        'analytical_gamma1': moments['gamma1'],
        'analytical_gamma2': moments['gamma2'],
        'cf_quantile': cf_quantile,
        'cf_valid': cf_valid,
        'cf_warning': cf_warning,
        'mc_mean': mc_stats['mean'],
        'mc_std': mc_stats['std'],
        'mc_skewness': mc_stats['skewness'],
        'mc_kurtosis': mc_stats['kurtosis'],
        'mc_quantile': mc_quantile,
        'mc_p75': mc_stats['p75'],
        'mc_p90': mc_stats['p90'],
        'mc_p95': mc_stats['p95'],
        'mc_p99': mc_stats['p99'],
        'relative_error': rel_error,
        'absolute_error': cf_quantile - mc_quantile
    }


# =============================================================================
# Risk Register Scenarios
# =============================================================================

def create_baseline_scenario() -> List[Risk]:
    """
    Scenario 1: Baseline
    n=30 risks, 25 threats + 5 opportunities
    Moderate uncertainty, typical project
    """
    risks = []
    
    # 25 Threats
    threat_data = [
        ("T01", "Scope creep", 0.40, 15000, 45000),
        ("T02", "Key resource unavailable", 0.30, 20000, 60000),
        ("T03", "Technology integration issues", 0.35, 25000, 80000),
        ("T04", "Vendor delays", 0.45, 10000, 35000),
        ("T05", "Requirements changes", 0.50, 12000, 40000),
        ("T06", "Testing delays", 0.35, 8000, 25000),
        ("T07", "Infrastructure problems", 0.20, 30000, 90000),
        ("T08", "Security vulnerabilities", 0.25, 20000, 70000),
        ("T09", "Data migration issues", 0.30, 15000, 50000),
        ("T10", "Training delays", 0.40, 5000, 18000),
        ("T11", "Regulatory compliance", 0.15, 40000, 120000),
        ("T12", "Communication failures", 0.35, 8000, 28000),
        ("T13", "Budget constraints", 0.25, 25000, 75000),
        ("T14", "Quality issues", 0.30, 18000, 55000),
        ("T15", "Stakeholder conflicts", 0.20, 12000, 40000),
        ("T16", "Documentation gaps", 0.45, 5000, 15000),
        ("T17", "Performance issues", 0.25, 22000, 65000),
        ("T18", "Dependencies on other projects", 0.35, 15000, 50000),
        ("T19", "Change management resistance", 0.30, 10000, 35000),
        ("T20", "Hardware failures", 0.15, 25000, 80000),
        ("T21", "Software bugs", 0.40, 8000, 30000),
        ("T22", "Network issues", 0.20, 12000, 40000),
        ("T23", "Licensing problems", 0.25, 10000, 35000),
        ("T24", "Environmental factors", 0.10, 30000, 100000),
        ("T25", "Legal disputes", 0.10, 50000, 150000),
    ]
    
    for id_, name, prob, q10, q90 in threat_data:
        risks.append(Risk(id_, name, prob, q10, q90, is_threat=True))
    
    # 5 Opportunities
    opportunity_data = [
        ("O01", "Early vendor delivery", 0.20, 10000, 30000),
        ("O02", "Resource efficiency gains", 0.25, 8000, 25000),
        ("O03", "Reuse of existing components", 0.30, 15000, 45000),
        ("O04", "Favorable exchange rates", 0.15, 5000, 20000),
        ("O05", "Process improvements", 0.20, 12000, 35000),
    ]
    
    for id_, name, prob, q10, q90 in opportunity_data:
        risks.append(Risk(id_, name, prob, q10, q90, is_threat=False))
    
    return risks


def create_small_scenario() -> List[Risk]:
    """
    Scenario 2: Small register
    n=10 risks, 8 threats + 2 opportunities
    """
    risks = []
    
    threat_data = [
        ("T01", "Scope creep", 0.45, 20000, 60000),
        ("T02", "Resource issues", 0.35, 25000, 75000),
        ("T03", "Technical debt", 0.40, 30000, 90000),
        ("T04", "Vendor problems", 0.30, 15000, 50000),
        ("T05", "Quality issues", 0.35, 20000, 65000),
        ("T06", "Timeline pressure", 0.50, 18000, 55000),
        ("T07", "Integration challenges", 0.25, 35000, 100000),
        ("T08", "Stakeholder changes", 0.20, 25000, 80000),
    ]
    
    for id_, name, prob, q10, q90 in threat_data:
        risks.append(Risk(id_, name, prob, q10, q90, is_threat=True))
    
    opportunity_data = [
        ("O01", "Team synergies", 0.25, 15000, 45000),
        ("O02", "Tool automation", 0.30, 10000, 35000),
    ]
    
    for id_, name, prob, q10, q90 in opportunity_data:
        risks.append(Risk(id_, name, prob, q10, q90, is_threat=False))
    
    return risks


def create_large_scenario() -> List[Risk]:
    """
    Scenario 3: Large register
    n=100 risks, 85 threats + 15 opportunities
    Typical large infrastructure project
    """
    np.random.seed(123)
    risks = []
    
    # Generate 85 threats with varying characteristics
    for i in range(85):
        prob = np.random.uniform(0.10, 0.50)
        base_impact = np.random.uniform(5000, 50000)
        q10 = base_impact * np.random.uniform(0.6, 0.9)
        q90 = base_impact * np.random.uniform(2.0, 4.0)
        risks.append(Risk(f"T{i+1:02d}", f"Threat {i+1}", prob, q10, q90, is_threat=True))
    
    # Generate 15 opportunities
    for i in range(15):
        prob = np.random.uniform(0.15, 0.35)
        base_impact = np.random.uniform(5000, 30000)
        q10 = base_impact * np.random.uniform(0.6, 0.9)
        q90 = base_impact * np.random.uniform(2.0, 3.5)
        risks.append(Risk(f"O{i+1:02d}", f"Opportunity {i+1}", prob, q10, q90, is_threat=False))
    
    return risks


def create_heterogeneous_scenario() -> List[Risk]:
    """
    Scenario 4: High heterogeneity
    One dominant risk with high probability and impact
    """
    risks = []
    
    # One dominant threat
    risks.append(Risk("T01", "Major regulatory change", 0.80, 200000, 600000, is_threat=True))
    
    # 19 smaller threats
    small_threats = [
        ("T02", "Minor scope changes", 0.35, 5000, 15000),
        ("T03", "Resource turnover", 0.30, 8000, 25000),
        ("T04", "Tool issues", 0.40, 3000, 10000),
        ("T05", "Communication gaps", 0.45, 4000, 12000),
        ("T06", "Testing overruns", 0.35, 6000, 18000),
        ("T07", "Documentation delays", 0.50, 2000, 8000),
        ("T08", "Training needs", 0.30, 5000, 15000),
        ("T09", "Meeting overhead", 0.60, 2000, 6000),
        ("T10", "Rework cycles", 0.40, 7000, 22000),
        ("T11", "Approval delays", 0.35, 4000, 14000),
        ("T12", "Environment issues", 0.25, 8000, 25000),
        ("T13", "Dependency waits", 0.30, 6000, 20000),
        ("T14", "Code reviews", 0.45, 3000, 10000),
        ("T15", "Bug fixes", 0.50, 5000, 16000),
        ("T16", "Performance tuning", 0.25, 10000, 30000),
        ("T17", "Security patches", 0.20, 8000, 25000),
        ("T18", "Data cleanup", 0.35, 4000, 12000),
        ("T19", "Report generation", 0.40, 2000, 7000),
    ]
    
    for id_, name, prob, q10, q90 in small_threats:
        risks.append(Risk(id_, name, prob, q10, q90, is_threat=True))
    
    # 5 opportunities
    opportunities = [
        ("O01", "Process reuse", 0.25, 8000, 25000),
        ("O02", "Early completion", 0.20, 10000, 30000),
        ("O03", "Volume discounts", 0.30, 5000, 15000),
        ("O04", "Skill transfer", 0.25, 6000, 18000),
        ("O05", "Automation gains", 0.20, 12000, 35000),
    ]
    
    for id_, name, prob, q10, q90 in opportunities:
        risks.append(Risk(id_, name, prob, q10, q90, is_threat=False))
    
    return risks


def create_balanced_scenario() -> List[Risk]:
    """
    Scenario 5: Balanced portfolio
    Equal threats and opportunities with similar magnitudes
    """
    risks = []
    
    # 15 threats
    for i in range(15):
        prob = 0.25 + (i % 5) * 0.05  # 0.25 to 0.45
        base = 15000 + i * 2000
        q10 = base * 0.7
        q90 = base * 2.5
        risks.append(Risk(f"T{i+1:02d}", f"Threat {i+1}", prob, q10, q90, is_threat=True))
    
    # 15 opportunities (similar magnitudes)
    for i in range(15):
        prob = 0.20 + (i % 5) * 0.05  # 0.20 to 0.40
        base = 12000 + i * 2000
        q10 = base * 0.7
        q90 = base * 2.5
        risks.append(Risk(f"O{i+1:02d}", f"Opportunity {i+1}", prob, q10, q90, is_threat=False))
    
    return risks


def create_high_uncertainty_scenario() -> List[Risk]:
    """
    Scenario 6: High uncertainty
    Same as baseline but with doubled uncertainty (wider q10-q90 range)
    """
    risks = []
    
    # 25 Threats with high uncertainty
    threat_data = [
        ("T01", "Scope creep", 0.40, 10000, 90000),
        ("T02", "Key resource unavailable", 0.30, 12000, 120000),
        ("T03", "Technology integration issues", 0.35, 15000, 160000),
        ("T04", "Vendor delays", 0.45, 6000, 70000),
        ("T05", "Requirements changes", 0.50, 8000, 80000),
        ("T06", "Testing delays", 0.35, 5000, 50000),
        ("T07", "Infrastructure problems", 0.20, 18000, 180000),
        ("T08", "Security vulnerabilities", 0.25, 12000, 140000),
        ("T09", "Data migration issues", 0.30, 9000, 100000),
        ("T10", "Training delays", 0.40, 3000, 36000),
        ("T11", "Regulatory compliance", 0.15, 25000, 240000),
        ("T12", "Communication failures", 0.35, 5000, 56000),
        ("T13", "Budget constraints", 0.25, 15000, 150000),
        ("T14", "Quality issues", 0.30, 11000, 110000),
        ("T15", "Stakeholder conflicts", 0.20, 7000, 80000),
        ("T16", "Documentation gaps", 0.45, 3000, 30000),
        ("T17", "Performance issues", 0.25, 13000, 130000),
        ("T18", "Dependencies on other projects", 0.35, 9000, 100000),
        ("T19", "Change management resistance", 0.30, 6000, 70000),
        ("T20", "Hardware failures", 0.15, 15000, 160000),
        ("T21", "Software bugs", 0.40, 5000, 60000),
        ("T22", "Network issues", 0.20, 7000, 80000),
        ("T23", "Licensing problems", 0.25, 6000, 70000),
        ("T24", "Environmental factors", 0.10, 18000, 200000),
        ("T25", "Legal disputes", 0.10, 30000, 300000),
    ]
    
    for id_, name, prob, q10, q90 in threat_data:
        risks.append(Risk(id_, name, prob, q10, q90, is_threat=True))
    
    # 5 Opportunities with high uncertainty
    opportunity_data = [
        ("O01", "Early vendor delivery", 0.20, 6000, 60000),
        ("O02", "Resource efficiency gains", 0.25, 5000, 50000),
        ("O03", "Reuse of existing components", 0.30, 9000, 90000),
        ("O04", "Favorable exchange rates", 0.15, 3000, 40000),
        ("O05", "Process improvements", 0.20, 7000, 70000),
    ]
    
    for id_, name, prob, q10, q90 in opportunity_data:
        risks.append(Risk(id_, name, prob, q10, q90, is_threat=False))
    
    return risks


# =============================================================================
# Main Analysis
# =============================================================================

def run_full_analysis():
    """Run complete analysis across all scenarios."""
    
    scenarios = {
        'Baseline (n=30)': create_baseline_scenario(),
        'Small (n=10)': create_small_scenario(),
        'Large (n=100)': create_large_scenario(),
        'High heterogeneity': create_heterogeneous_scenario(),
        'Balanced portfolio': create_balanced_scenario(),
        'High uncertainty': create_high_uncertainty_scenario(),
    }
    
    results = []
    full_results = []

    print("=" * 80)
    print("NUMERICAL VALIDATION: Cornish-Fisher vs Monte Carlo")
    print("=" * 80)
    print()

    for scenario_name, risks in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")
        
        comparison = compare_methods(risks, alpha=0.90, n_simulations=1_000_000)
        
        print(f"\nRisk Register: {comparison['n_risks']} risks "
              f"({comparison['n_threats']} threats, {comparison['n_opportunities']} opportunities)")
        
        print(f"\n--- Analytical Results ---")
        print(f"Mean:              {comparison['analytical_mean']:>12,.0f}")
        print(f"Std Dev:           {comparison['analytical_std']:>12,.0f}")
        print(f"Skewness (γ₁):     {comparison['analytical_gamma1']:>12.3f}")
        print(f"Excess Kurt (γ₂):  {comparison['analytical_gamma2']:>12.3f}")
        print(f"CF P90:            {comparison['cf_quantile']:>12,.0f}")
        print(f"CF Valid:          {comparison['cf_valid']}")
        if comparison['cf_warning'] != 'OK':
            print(f"CF Warning:        {comparison['cf_warning']}")
        
        print(f"\n--- Monte Carlo Results (n=1,000,000) ---")
        print(f"Mean:              {comparison['mc_mean']:>12,.0f}")
        print(f"Std Dev:           {comparison['mc_std']:>12,.0f}")
        print(f"Skewness:          {comparison['mc_skewness']:>12.3f}")
        print(f"Excess Kurtosis:   {comparison['mc_kurtosis']:>12.3f}")
        print(f"MC P90:            {comparison['mc_quantile']:>12,.0f}")
        
        print(f"\n--- Comparison ---")
        print(f"Absolute Error:    {comparison['absolute_error']:>12,.0f}")
        print(f"Relative Error:    {comparison['relative_error']:>12.2%}")
        
        # Store results
        results.append({
            'Scenario': scenario_name,
            'n': comparison['n_risks'],
            'γ₁': comparison['analytical_gamma1'],
            'γ₂': comparison['analytical_gamma2'],
            'MC P90': comparison['mc_quantile'],
            'CF P90': comparison['cf_quantile'],
            'Rel. Error': comparison['relative_error']
        })
        full_results.append({
            'Scenario': scenario_name,
            'n': comparison['n_risks'],
            'Threats': comparison['n_threats'],
            'Opportunities': comparison['n_opportunities'],
            'Mean_Analytical': round(comparison['analytical_mean'], 2),
            'Std_Analytical': round(comparison['analytical_std'], 2),
            'Skewness_gamma1': round(comparison['analytical_gamma1'], 4),
            'Excess_Kurtosis_gamma2': round(comparison['analytical_gamma2'], 4),
            'Mean_MC': round(comparison['mc_mean'], 2),
            'Std_MC': round(comparison['mc_std'], 2),
            'Skewness_MC': round(comparison['mc_skewness'], 4),
            'Kurtosis_MC': round(comparison['mc_kurtosis'], 4),
            'P75_MC': round(comparison['mc_p75'], 2),
            'P90_MC': round(comparison['mc_quantile'], 2),
            'P95_MC': round(comparison['mc_p95'], 2),
            'P99_MC': round(comparison['mc_p99'], 2),
            'P90_CF': round(comparison['cf_quantile'], 2),
            'Absolute_Error': round(comparison['absolute_error'], 2),
            'Relative_Error_Pct': round(comparison['relative_error'] * 100, 2),
            'CF_Valid': comparison['cf_valid'],
            'Warning': comparison['cf_warning'],
        })
    
    # Summary table
    print("\n\n")
    print("=" * 80)
    print("SUMMARY TABLE (for paper)")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    df_results['MC P90'] = df_results['MC P90'].apply(lambda x: f"{x:,.0f}")
    df_results['CF P90'] = df_results['CF P90'].apply(lambda x: f"{x:,.0f}")
    df_results['γ₁'] = df_results['γ₁'].apply(lambda x: f"{x:.2f}")
    df_results['γ₂'] = df_results['γ₂'].apply(lambda x: f"{x:.2f}")
    df_results['Rel. Error'] = df_results['Rel. Error'].apply(lambda x: f"{x:+.1%}")
    
    print(df_results.to_string(index=False))
    
    # LaTeX table
    print("\n\n--- LaTeX Table ---\n")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Comparison of Cornish-Fisher approximation versus Monte Carlo simulation}")
    print(r"\label{tab:validation}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Scenario & $\gamma_1$ & $\gamma_2$ & MC P90 & CF P90 & Rel. Error \\")
    print(r"\midrule")
    
    for _, row in pd.DataFrame(results).iterrows():
        print(f"{row['Scenario']} & {row['γ₁']:.2f} & {row['γ₂']:.2f} & "
              f"{row['MC P90']:,.0f} & {row['CF P90']:,.0f} & {row['Rel. Error']:+.1%} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Save results to CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(full_results).to_csv(RESULTS_DIR / 'validation_results_full.csv', index=False)
    summary_df = pd.DataFrame({
        'Scenario': [r['Scenario'] for r in results],
        'n': [r['n'] for r in results],
        'gamma1': [round(r['γ₁'], 2) for r in results],
        'gamma2': [round(r['γ₂'], 2) for r in results],
        'MC_P90': [int(round(r['MC P90'])) for r in results],
        'CF_P90': [int(round(r['CF P90'])) for r in results],
        'Relative_Error_Pct': [round(r['Rel. Error'] * 100, 1) for r in results],
    })
    summary_df.to_csv(RESULTS_DIR / 'summary_table.csv', index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")

    return results


def export_risk_registers():
    """Export risk registers to CSV for documentation."""
    
    scenarios = {
        'baseline': create_baseline_scenario(),
        'small': create_small_scenario(),
        'large': create_large_scenario(),
        'heterogeneous': create_heterogeneous_scenario(),
        'balanced': create_balanced_scenario(),
        'high_uncertainty': create_high_uncertainty_scenario(),
    }
    
    for name, risks in scenarios.items():
        rows = []
        for r in risks:
            rows.append({
                'ID': r.id,
                'Name': r.name,
                'Probability': r.probability,
                'P10': r.q10,
                'P90': r.q90,
                'Type': 'Threat' if r.is_threat else 'Opportunity',
                'mu': r.mu,
                'sigma': r.sigma,
                'E[D]': r.lognormal_mean,
            })
        
        df = pd.DataFrame(rows)
        filepath = DATA_DIR / f'risk_register_{name}.csv'
        df.to_csv(filepath, index=False)
        print(f"Exported: {filepath}")


def analyze_sensitivity():
    """
    Sensitivity analysis: How does error vary with number of risks?
    """
    print("\n\n")
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: Error vs Number of Risks")
    print("=" * 80)
    
    np.random.seed(456)
    
    n_values = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    results = []
    
    for n in n_values:
        # Generate n risks (80% threats, 20% opportunities)
        risks = []
        n_threats = int(n * 0.8)
        n_opps = n - n_threats
        
        for i in range(n_threats):
            prob = np.random.uniform(0.15, 0.45)
            base = np.random.uniform(10000, 40000)
            q10 = base * np.random.uniform(0.6, 0.85)
            q90 = base * np.random.uniform(2.2, 3.5)
            risks.append(Risk(f"T{i+1}", f"Threat {i+1}", prob, q10, q90, is_threat=True))
        
        for i in range(n_opps):
            prob = np.random.uniform(0.15, 0.35)
            base = np.random.uniform(8000, 25000)
            q10 = base * np.random.uniform(0.6, 0.85)
            q90 = base * np.random.uniform(2.0, 3.0)
            risks.append(Risk(f"O{i+1}", f"Opp {i+1}", prob, q10, q90, is_threat=False))
        
        comparison = compare_methods(risks, alpha=0.90, n_simulations=50000)
        
        results.append({
            'n': n,
            'gamma1': comparison['analytical_gamma1'],
            'gamma2': comparison['analytical_gamma2'],
            'mc_p90': comparison['mc_quantile'],
            'cf_p90': comparison['cf_quantile'],
            'rel_error': comparison['relative_error'],
        })

        print(f"n={n:3d}: γ₁={comparison['analytical_gamma1']:6.3f}, "
              f"γ₂={comparison['analytical_gamma2']:6.3f}, "
              f"error={comparison['relative_error']:+6.2%}")

    # Save sensitivity results to CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({
        'n': [r['n'] for r in results],
        'gamma1': [round(r['gamma1'], 3) for r in results],
        'gamma2': [round(r['gamma2'], 3) for r in results],
        'MC_P90': [int(round(r['mc_p90'])) for r in results],
        'CF_P90': [int(round(r['cf_p90'])) for r in results],
        'Relative_Error_Pct': [round(r['rel_error'] * 100, 2) for r in results],
    }).to_csv(RESULTS_DIR / 'sensitivity_analysis.csv', index=False)
    print(f"Sensitivity results saved to {RESULTS_DIR}/sensitivity_analysis.csv")

    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run main analysis
    results = run_full_analysis()
    
    # Export risk registers
    print("\n\nExporting risk registers to CSV...")
    export_risk_registers()
    
    # Sensitivity analysis
    sensitivity = analyze_sensitivity()
    
    print("\n\nAnalysis complete!")
