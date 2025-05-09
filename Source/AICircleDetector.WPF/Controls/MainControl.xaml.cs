using AICircleDetector.WPF.ViewModels;
using System.Windows.Controls;

namespace AICircleDetector.WPF.Controls
{

    /// <summary>
    /// Interaktionslogik für WelcomeControl.xaml
    /// </summary>
    public partial class MainControl : UserControl
    {

        private MainControlViewModel ViewModel = new();

        public MainControl()
        {
            InitializeComponent();
            DataContext = ViewModel;
        }

    }
}
