namespace AICircleDetector.WPF.Converter
{
    using System;
    using System.Windows;
    using System.Windows.Data;

    public class BoolToVisibilityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (value is bool boolValue)
            {
                bool invert = parameter as string == "invert"; // Check if parameter is 'invert'
                if (invert)
                {
                    // Invert the visibility
                    return boolValue ? Visibility.Collapsed : Visibility.Visible;
                }
                else
                {
                    // Default visibility
                    return boolValue ? Visibility.Visible : Visibility.Collapsed;
                }
            }
            return Visibility.Collapsed; // Default if the value is not a boolean
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (value is Visibility visibilityValue)
            {
                // Convert Visibility back to bool
                return visibilityValue == Visibility.Visible;
            }
            return false; // Default to false if not Visibility
        }
    }

}
