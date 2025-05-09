using AICircleDetector.WPF.ViewModels;
using System.Windows.Controls;


namespace AICircleDetector.WPF.Controls
{
    /// <summary>
    /// Interaktionslogik für Footer.xaml
    /// </summary>
    public partial class Footer : UserControl
    {
        public Footer()
        {
            InitializeComponent();
            DataContext = new FooterViewModel();
        }
    }
    
}