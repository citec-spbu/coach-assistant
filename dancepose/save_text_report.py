"""
Saves text report to file
"""
import json
from pathlib import Path
import sys

def save_report(video_folder):
    """Saves report to text file"""
    
    analysis_file = Path(video_folder) / "analysis_result.json"
    output_file = Path(video_folder) / "REPORT.txt"
    
    if not analysis_file.exists():
        print(f"Error: file {analysis_file} not found")
        return
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Form report
    report = []
    report.append("=" * 80)
    report.append("DANCE ANALYSIS RESULTS")
    report.append("=" * 80)
    report.append("")
    report.append(f"Figure: {data['classification']['figure']}")
    report.append(f"Confidence: {data['classification']['confidence']:.1%}")
    report.append(f"")
    report.append(f"OVERALL SCORE: {data['scores']['overall']:.1f}/100")
    report.append("")
    report.append("DETAILED SCORES:")
    
    status_mapping = {
        'excellent': 'excellent',
        'good': 'good',
        'fair': 'fair',
        'needs improvement': 'needs improvement'
    }
    
    for key in ['technique', 'timing', 'balance', 'dynamics', 'posture']:
        score = data['scores'][key]['score']
        status = data['scores'][key]['status']
        bar = '=' * int(score/5) + '-' * (20 - int(score/5))
        report.append(f"  {key.capitalize():<12} [{bar}] {score:5.1f}/100  ({status_mapping[status]})")
    
    report.append("")
    report.append(f"ERRORS FOUND: {data['total_errors']}")
    
    if data['errors']:
        report.append("")
        report.append("TOP 5 ERRORS:")
        
        for i, error in enumerate(data['errors'][:5], 1):
            category = error.get('category', 'unknown').upper()
            severity = error.get('severity', 'medium')
            
            report.append("")
            report.append(f"  {i}. Time: {error['timestamp']:.1f}s")
            report.append(f"     Category: {category}")
            report.append(f"     Description: {error.get('issue', error.get('description', 'unknown'))}")
            report.append(f"     Severity: {severity}")
    
    report.append("")
    report.append("=" * 80)
    report.append("DETAILED INFORMATION:")
    report.append("=" * 80)
    
    # Technique
    report.append("")
    report.append("TECHNIQUE:")
    if 'details' in data['scores']['technique']:
        for joint, data_joint in data['scores']['technique']['details'].items():
            score = data_joint['score']
            deviation = data_joint['avg_deviation']
            report.append(f"  {joint:<20} {score:5.1f}/100  (deviation: {deviation:.2f})")
    
    # Timing
    report.append("")
    report.append("TIMING:")
    timing = data['scores']['timing']['details']
    report.append(f"  Average velocity:     {timing['avg_velocity']:.1f} px/frame")
    report.append(f"  Average jerk:         {timing['avg_jerk']:.1f}")
    report.append(f"  Max jerk:             {timing['max_jerk']:.1f}")
    report.append(f"  Smoothness:           {timing['smoothness']:.1%}")
    
    # Balance
    report.append("")
    report.append("BALANCE:")
    balance = data['scores']['balance']['details']
    report.append(f"  Average deviation:    {balance['avg_deviation']:.1f} px")
    report.append(f"  Max deviation:        {balance['max_deviation']:.1f} px")
    report.append(f"  Stability:            {balance['stability']:.1%}")
    report.append(f"  Symmetry:             {balance['symmetry']:.1%}")
    
    # Dynamics
    report.append("")
    report.append("DYNAMICS:")
    dynamics = data['scores']['dynamics']['details']
    report.append(f"  Average amplitude:    {dynamics['avg_amplitude']:.1f} px")
    report.append(f"  Total energy:         {dynamics['energy']:.0f}")
    report.append(f"  Energy per frame:     {dynamics['energy_per_frame']:.1f}")
    report.append(f"  Contrast:             {dynamics['contrast']:.1%}")
    
    # Posture
    report.append("")
    report.append("POSTURE:")
    posture = data['scores']['posture']['details']
    report.append(f"  Spine angle:          {posture['spine_angle']:.1f} degrees")
    report.append(f"  Head height:          {posture['head_height']:.2f}")
    report.append(f"  Chest openness:       {posture['chest_openness']:.1%}")
    report.append(f"  Spine score:          {posture['spine_score']:.1f}/100")
    report.append(f"  Head score:           {posture['head_score']:.1f}/100")
    report.append(f"  Chest score:          {posture['chest_score']:.1f}/100")
    
    report.append("")
    report.append("=" * 80)
    report.append("ANALYSIS COMPLETE")
    report.append("=" * 80)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved: {output_file}")
    
    # Open file
    import os
    os.startfile(str(output_file.absolute()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python save_text_report.py <video_folder>")
    else:
        save_report(sys.argv[1])
