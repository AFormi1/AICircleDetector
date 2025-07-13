using AICircleDetector.AI;
using AICircleDetector.WPF.Converter;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using OneOf.Types;
using System.Diagnostics;
using System.IO;
using System.Net.WebSockets;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace AICircleDetector.WPF.ViewModels
{

    public partial class MainControlViewModel : BaseViewModel
    {

        private Canvas CanvasFromUI;
        private Image ImageFromUI;
        private List<Circle> PredictedCircles;

        private int TrainingSetsCount = 1;

        private CancellationTokenSource CancellationTokenSource = new CancellationTokenSource();
        private CancellationToken CancelToken;

        [ObservableProperty]
        private ImageSource imageToDisplay;

        private string imageURL = string.Empty;

        [ObservableProperty]
        private string imageCount = "200";

        [ObservableProperty]
        private bool consoleEnabled = true;

        [ObservableProperty]
        private bool dataButtonEnabled = true;

        [ObservableProperty]
        private bool isBusyDataCreation = false;

        [ObservableProperty]
        private bool trainingButtonEnabled = true;

        [ObservableProperty]
        private bool isBusyTraining = false;

        [ObservableProperty]
        private string consoleText = string.Empty;




        public MainControlViewModel()
        {
            ConsoleBindingWriter writer = new ConsoleBindingWriter(AppendConsoleLine);
            Console.SetOut(writer);
        }


        private void AppendConsoleLine(string line)
        {
            // Run on UI thread if necessary
            Application.Current.Dispatcher.Invoke(() =>
            {
                ConsoleText += line + Environment.NewLine;
            });
        }

        [RelayCommand]
        public async Task CreateTrainingData()
        {
            DataButtonEnabled = false;
            IsBusyDataCreation = true;

            bool result = false;

            EnableConsole();

            for (int i = 0; i < TrainingSetsCount; i++)
            {
                _ = int.TryParse(ImageCount, out int imageCount);

                if (imageCount == 0)
                {
                    imageCount = 200;
                    ImageCount = "200";
                }
                await Task.Run(() =>
                {
                    result = AI.TrainingDataBuilder.CreateTrainingData(imageCount: imageCount);

                    string msg = "";
                    if (result)
                    {
                        msg = $"Testfiles have been created successfully to\r\n{System.IO.Path.Combine(Environment.CurrentDirectory, AIConfig.TrainingFolderName)}";
                        MessageBox.Show(msg, "Result", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                    else
                    {
                        msg = $"An error occurred while creating the testfiles!";
                        MessageBox.Show(msg, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    }

                    Console.WriteLine(msg);

                    DataButtonEnabled = true;
                    IsBusyDataCreation = false;
                });
            }
        }


        [RelayCommand]
        public async Task TrainAI()
        {
            List<AI.AIResult> results = new();

            ConsoleText = string.Empty;

            TrainingButtonEnabled = false;
            IsBusyTraining = true;

            CancellationTokenSource = new CancellationTokenSource();
            CancelToken = CancellationTokenSource.Token;

            string result = string.Empty;

            try
            {
                OpenFolderDialog folderDialog = new OpenFolderDialog
                {
                    Title = "Select Folder"
                };

                if (folderDialog.ShowDialog() == true)
                {
                    EnableConsole();

                    string folderName = folderDialog.FolderName;

                    await Task.Run(() =>
                    {
                        Stopwatch stopwatch = new Stopwatch();
                        stopwatch.Start();

                        result = AI.Trainer.Train(folderDialog.FolderName);

                        stopwatch.Stop();

                        result += $"\r\nTraining took {stopwatch.Elapsed.TotalSeconds:F0} s";

                        if (result.Contains("completed"))
                            MessageBox.Show(result, "Training completed", MessageBoxButton.OK, MessageBoxImage.Information);
                        else
                            MessageBox.Show(result, "Training failed", MessageBoxButton.OK, MessageBoxImage.Error);

                        Console.WriteLine(result);

                        DataButtonEnabled = true;
                        IsBusyDataCreation = false;
                    });
                }
                else
                {
                    throw new OperationCanceledException("Folder selection was cancelled.");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"An error occurred: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                TrainingButtonEnabled = true;
                IsBusyTraining = false;
            }
        }

        private void EnableConsole()
        {
            if (ConsoleEnabled)
            {
                Console.SetOut(new ConsoleBindingWriter(AppendConsoleLine));
            }
            else
            {
                Console.SetOut(new ConsoleBindingWriter(AppendConsoleLine));
                Console.WriteLine("Logging disabled due to speed up the process");
                Console.SetOut(TextWriter.Null);
            }
        }



        [RelayCommand]
        public void OpenImage()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.gif",
                Title = "Select an Image"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                string filePath = openFileDialog.FileName;
                ImageToDisplay = new BitmapImage(new Uri(filePath));

                imageURL = openFileDialog.FileName;

                PredictedCircles = null;

                // Delay CanvasOverlay until UI finishes rendering
                Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                {
                    Task.Delay(100).Wait();
                    CanvasOverlay();

                }), DispatcherPriority.Loaded);
            }
        }

        [RelayCommand]
        public async Task DetectCircle()
        {
            if (string.IsNullOrEmpty(imageURL))
            {
                MessageBox.Show($"An image must be selected before we can make a prediction!", "No Image selected", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            ConsoleText = string.Empty;

            try
            {
                EnableConsole();

                await Task.Run(() =>
                {
                    Stopwatch stopwatch = new Stopwatch();
                    stopwatch.Start();

                    PredictionResult result;

                    result = AI.Predictor.Predict(imageURL);

                    stopwatch.Stop();

                    result.Result += $"\r\nPrediction took {stopwatch.Elapsed.TotalMilliseconds:F0} ms";

                    if (result.Result.Contains("completed"))
                        MessageBox.Show(result.Result, "Prediction finished", MessageBoxButton.OK, MessageBoxImage.Information);
                    else
                        MessageBox.Show(result.Result, "Prediction failed", MessageBoxButton.OK, MessageBoxImage.Error);

                    Console.WriteLine(result);

                    DataButtonEnabled = true;
                    IsBusyDataCreation = false;

                    //Draw the circles on the canvas
                    PredictedCircles = result.Circles;

                    // Delay CanvasOverlay until UI finishes rendering
                    Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        CanvasOverlay();

                    }), DispatcherPriority.Loaded);

                });

            }

            catch (Exception ex)
            {
                MessageBox.Show($"An error occurred: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                TrainingButtonEnabled = true;
                IsBusyTraining = false;
            }
        }
        [RelayCommand]
        public void CancelTrainAI()
        {
            CancellationTokenSource.Cancel();

            ImageToDisplay = null;
            imageURL = string.Empty;

            TrainingButtonEnabled = true;
            IsBusyTraining = false;

        }


        public void CanvasOverlay()
        {
            if (CanvasFromUI == null || ImageFromUI == null || ImageToDisplay == null) return;

            CanvasFromUI.Children.Clear();

            double width = ImageFromUI.ActualWidth;
            double height = ImageFromUI.ActualHeight;

            CanvasFromUI.Width = width;
            CanvasFromUI.Height = height;

            if (width == 0 || height == 0) return;

            int linesCount = 10;

            for (int i = 0; i <= linesCount; i++)
            {
                double fraction = i / (double)linesCount;

                // Vertical line
                var vLine = new System.Windows.Shapes.Line
                {
                    X1 = fraction * width,
                    Y1 = 0,
                    X2 = fraction * width,
                    Y2 = height,
                    Stroke = System.Windows.Media.Brushes.Gray,
                    StrokeThickness = 1
                };
                CanvasFromUI.Children.Add(vLine);

                // Vertical line label at top (e.g., 0.0, 0.1,...)
                var vLabel = new TextBlock
                {
                    Text = fraction.ToString("0.0"),
                    Foreground = System.Windows.Media.Brushes.Black,
                    FontSize = 10
                };
                Canvas.SetLeft(vLabel, fraction * width);
                Canvas.SetTop(vLabel, 0);
                CanvasFromUI.Children.Add(vLabel);

                // Horizontal line
                var hLine = new System.Windows.Shapes.Line
                {
                    X1 = 0,
                    Y1 = fraction * height,
                    X2 = width,
                    Y2 = fraction * height,
                    Stroke = System.Windows.Media.Brushes.Gray,
                    StrokeThickness = 1
                };
                CanvasFromUI.Children.Add(hLine);

                // Horizontal line label at left side
                var hLabel = new TextBlock
                {
                    Text = fraction.ToString("0.0"),
                    Foreground = System.Windows.Media.Brushes.Black,
                    FontSize = 10
                };
                Canvas.SetLeft(hLabel, 0);
                Canvas.SetTop(hLabel, fraction * height);
                CanvasFromUI.Children.Add(hLabel);
            }

            if (PredictedCircles != null)
            {
                for (int i = 0; i < PredictedCircles.Count; i++)
                {
                    var circle = PredictedCircles[i];

                    // Skip invalid bounds
                    if (circle.XMin < 0 || circle.YMin < 0 ||
                        circle.XMax > 1 || circle.YMax > 1 ||
                        circle.XMin >= circle.XMax || circle.YMin >= circle.YMax)
                    {
                        continue;
                    }

                    double xMin = circle.XMin * width;
                    double yMin = circle.YMin * height;
                    double xMax = circle.XMax * width;
                    double yMax = circle.YMax * height;

                    double centerX = (xMin + xMax) / 2;
                    double centerY = (yMin + yMax) / 2;
                    double radiusX = (xMax - xMin) / 2;
                    double radiusY = (yMax - yMin) / 2;
                               

                    // Draw ellipse
                    var ellipse = new Ellipse
                    {
                        Width = radiusX * 2,
                        Height = radiusY * 2,
                        Stroke = Brushes.Red,
                        StrokeThickness = 2,
                        Fill = Brushes.Transparent
                    };

                    Canvas.SetLeft(ellipse, centerX - radiusX);
                    Canvas.SetTop(ellipse, centerY - radiusY);
                    CanvasFromUI.Children.Add(ellipse);

                    // Draw index label at the center of the ellipse
                    var label = new TextBlock
                    {   
                        Text = $"[{circle.XMin:F3}|{circle.XMax:F3}|{circle.YMin:F3}|{circle.YMax:F3}]",
                        Foreground = Brushes.Red,
                        FontWeight = FontWeights.Bold,
                        FontSize = 12,
                        TextAlignment = TextAlignment.Center
                    };

                    // Measure the label size (optional for better centering)
                    label.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                    Size labelSize = label.DesiredSize;

                    Canvas.SetLeft(label, centerX - labelSize.Width / 2);
                    Canvas.SetTop(label, centerY - labelSize.Height / 2);
                    CanvasFromUI.Children.Add(label);
                }
            }


        }


        public void SetUpCanvas(Canvas canvas, Image image)
        {
            CanvasFromUI = canvas;
            ImageFromUI = image;
        }
    }

}
