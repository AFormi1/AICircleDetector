using CommunityToolkit.Mvvm.ComponentModel;
using System.Windows.Threading;

namespace AICircleDetector.WPF.ViewModels
{

    public partial class FooterViewModel : ObservableObject
    {
      
        [ObservableProperty]
        private string? logText;

        [ObservableProperty]
        private string? clockText;


        public FooterViewModel()
        {
 
            LogText = "System initialized";

            // Update ClockText every second
            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(100);
            timer.Tick += OnTimerElapsed;
            timer.Start();
        }

      
        // Timer callback method
        private void OnTimerElapsed(object? sender, EventArgs? e)
        {
            // Marshal the update to the UI thread using the Dispatcher
            ClockText = DateTime.Now.ToString("dd.MM.yyyy HH:mm:ss");
        }
    }
}
