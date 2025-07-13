using AICircleDetector.WPF.ViewModels;
using System.Drawing;
using System.Windows.Controls;
using System.Windows.Media;

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

            ViewModel.SetUpCanvas(OverlayCanvas, MainImage);
        }

        private void UserControl_SizeChanged(object sender, System.Windows.SizeChangedEventArgs e)
        {
            ViewModel.CanvasOverlay();
        }
      
    }
}
