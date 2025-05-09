using AICircleDetector.WPF.ViewModels;
using System.Windows.Controls;


namespace AICircleDetector.WPF.Controls
{
    /// <summary>
    /// Interaktionslogik für Header.xaml
    /// </summary>
    public partial class Header : UserControl
    {
        public Header()
        {
            InitializeComponent();
            DataContext = new HeaderViewModel();
        }
    }
    
}