using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace AICircleDetector.WPF.Converter
{
    public static class TextBoxAutoScroller
    {
        public static readonly DependencyProperty AutoScrollProperty =
            DependencyProperty.RegisterAttached(
                "AutoScroll",
                typeof(bool),
                typeof(TextBoxAutoScroller),
                new PropertyMetadata(false, OnAutoScrollChanged));

        public static bool GetAutoScroll(DependencyObject obj) => (bool)obj.GetValue(AutoScrollProperty);
        public static void SetAutoScroll(DependencyObject obj, bool value) => obj.SetValue(AutoScrollProperty, value);

        private static void OnAutoScrollChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is TextBox textBox)
            {
                if ((bool)e.NewValue)
                {
                    textBox.TextChanged += (s, args) =>
                    {
                        ScrollViewer scrollViewer = GetScrollViewer(textBox);
                        if (scrollViewer != null &&
                            scrollViewer.ExtentHeight > scrollViewer.ViewportHeight)
                        {
                            scrollViewer.ScrollToEnd();
                        }
                    };
                }
            }
        }

        private static ScrollViewer GetScrollViewer(DependencyObject depObj)
        {
            if (depObj is ScrollViewer)
                return (ScrollViewer)depObj;

            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(depObj); i++)
            {
                DependencyObject child = VisualTreeHelper.GetChild(depObj, i);
                ScrollViewer result = GetScrollViewer(child);
                if (result != null)
                    return result;
            }

            return null;
        }
    }
}
