using AICircleDetector.WPF.Controls;
using System.Windows.Controls;

namespace AICircleDetector.WPF.ViewModels
{
    public class NavigationService
    {
        private ContentControl? _contentControl;

        private object? _previousPage;

        // Initialize with the ContentControl instance
        public void Initialize(ContentControl contentControl)
        {
            _contentControl = contentControl;
            _previousPage = _contentControl.Content;
        }


        public void NavigateToHome()
        {
            if (_contentControl == null)
                throw new InvalidOperationException("NavigationService is not initialized with a ContentControl.");

            _previousPage = _contentControl.Content;
            _contentControl.Content = new MainControl(); ;
        }

       
        // Navigate to a specified UserControl
        public void NavigateTo(UserControl userControl)
        {
            if (_contentControl == null)
                throw new InvalidOperationException("NavigationService is not initialized with a ContentControl.");

            _previousPage = _contentControl.Content;
            _contentControl.Content = userControl;
        }

      

        public void NavigateBack()
        {
            if (_contentControl == null)
                throw new InvalidOperationException("NavigationService is not initialized with a ContentControl.");

            if (_previousPage == null)
                return;

            object tempPage = _contentControl.Content;
            _contentControl.Content = _previousPage;
            _previousPage = tempPage;
        }

    }


}
