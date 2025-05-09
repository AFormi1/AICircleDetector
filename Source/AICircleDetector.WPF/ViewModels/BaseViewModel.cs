using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AICircleDetector.WPF.ViewModels
{
    public partial class BaseViewModel : ObservableObject
    {
        [RelayCommand]
        private void Home()
        {
            MainPage.NavigationService.NavigateToHome();
        }


        [RelayCommand]
        public void NavigateBack()
        {
            MainPage.NavigationService.NavigateBack();
        }
    }
}
