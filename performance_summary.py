import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_performance_report():
    # Your evaluation results
    performance = {
        'Class': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR', 'Overall'],
        'Accuracy': [98.73, 37.78, 88.89, 0.00, 44.44, 81.20],
        'Samples': [236, 45, 153, 21, 45, 500],
        'Clinical_Importance': ['Critical', 'High', 'High', 'Critical', 'Critical', 'Overall']
    }
    
    df = pd.DataFrame(performance)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy by class
    colors = ['green', 'yellow', 'orange', 'red', 'red', 'blue']
    bars = ax1.bar(df['Class'], df['Accuracy'], color=colors, alpha=0.7)
    ax1.set_title('Model Accuracy by Diabetic Retinopathy Class', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, df['Accuracy']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Sample distribution
    ax2.pie(df['Samples'][:-1], labels=df['Class'][:-1], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Distribution in Test Set', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print recommendations
    print("🎯 CLINICAL PERFORMANCE ASSESSMENT")
    print("="*50)
    print("✅ STRENGTHS:")
    print("   - Excellent healthy patient identification (98.7%)")
    print("   - Good moderate case detection (88.9%)")
    print("   - Substantial overall agreement (Kappa: 0.706)")
    print("   - Ready for initial screening deployment")
    
    print("\n⚠️  AREAS FOR IMPROVEMENT:")
    print("   - Severe cases not detected (0%) - HIGH PRIORITY")
    print("   - Mild cases challenging (37.8%) - MODERATE PRIORITY")
    print("   - Class imbalance affecting performance")
    
    print("\n🚀 RECOMMENDATIONS:")
    print("   1. Add class weights to loss function")
    print("   2. Oversample Severe/Mild classes")
    print("   3. Try focal loss for imbalanced data")
    print("   4. Consider ensemble methods")
    print("   5. Add more Severe class samples if available")

if __name__ == "__main__":
    create_performance_report()