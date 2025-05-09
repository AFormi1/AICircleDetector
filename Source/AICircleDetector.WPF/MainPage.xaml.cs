using AICircleDetector.WPF.Controls;
using AICircleDetector.WPF.ViewModels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace AICircleDetector.WPF
{
    /// <summary>
    /// Interaktionslogik für Window1.xaml
    /// </summary>
    public partial class MainPage : Window
    {
        public static NavigationService NavigationService { get; } = new NavigationService();

        private MainPageViewModel ViewModel = new();

        public MainPage()
        {
            InitializeComponent();

            DataContext = ViewModel;

            NavigationService.Initialize(MainContent);

            // Set default content
            NavigationService.NavigateTo(new MainControl());

        }
    }
}
