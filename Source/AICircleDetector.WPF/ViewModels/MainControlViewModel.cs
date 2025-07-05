using AICircleDetector.AI;
using AICircleDetector.WPF.Converter;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace AICircleDetector.WPF.ViewModels
{

    public partial class MainControlViewModel : BaseViewModel
    {
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
        private bool validateButtonEnabled = true;

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
                await Task.Run(async () => result = await AI.TrainingDataBuilder.CreateTrainingData(imageCount: imageCount));
            }

            if (result)
                MessageBox.Show($"Testfiles have been created successfully to\r\n{Path.Combine(Environment.CurrentDirectory, AIConfig.TrainingFolderName)}", "Result", MessageBoxButton.OK, MessageBoxImage.Information);
            else
                MessageBox.Show($"An error occurred while creating the testfiles!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);

            DataButtonEnabled = true;
            IsBusyDataCreation = false;
        }


        [RelayCommand]
        public async Task TrainAI()
        {
            List<AI.AIResult> results = new();

            ConsoleText = string.Empty;

            TrainingButtonEnabled = false;
            ValidateButtonEnabled = false;
            IsBusyTraining = true;

            CancellationTokenSource = new CancellationTokenSource();
            CancelToken = CancellationTokenSource.Token;

            try
            {
                OpenFolderDialog folderDialog = new OpenFolderDialog
                {
                    Title = "Select Folder",
                    InitialDirectory = AppDomain.CurrentDomain.BaseDirectory
                };

                if (folderDialog.ShowDialog() == true)
                {
                    EnableConsole();

                    string folderName = folderDialog.FolderName;

                    Stopwatch stopwatch = new Stopwatch();
                    stopwatch.Start();

                    //results.Add(await Task.Run(() => AI.TrainerAndValidator.Train(CancelToken, folderName)));

                    stopwatch.Stop();

                    // Calculate the average loss and accuracy from the results
                    float res1 = results.Average(result => result.Loss);
                    float res2 = results.Average(result => result.MAE);

                    EnableConsole();

                    string message = $"Training complete!\nAverage Loss: {res1:F4}\nAverage MAE: {res2:F4}\r\nTime Elapsed: {stopwatch.Elapsed.TotalSeconds} [s]";

                    // Determine whether the overall result is a success or error
                    bool allSuccessful = results.All(result => result.Success);
                    string title = allSuccessful ? "Success" : "Error";
                    MessageBoxImage icon = allSuccessful ? MessageBoxImage.Information : MessageBoxImage.Error;

                    // Show success/error popup with averages
                    MessageBox.Show(message, title, MessageBoxButton.OK, icon);
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
                ValidateButtonEnabled = true;
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
                Console.WriteLine("Logging disabled due to speed up the process");
                Console.SetOut(TextWriter.Null);
            }
        }

        [RelayCommand]
        public async Task ValidateAI()
        {
            List<AI.AIResult> results = new();

            ConsoleText = string.Empty;

            TrainingButtonEnabled = false;
            ValidateButtonEnabled = false;
            IsBusyTraining = true;

            CancellationTokenSource = new CancellationTokenSource();
            CancelToken = CancellationTokenSource.Token;

            try
            {
                OpenFolderDialog folderDialog = new OpenFolderDialog
                {
                    Title = "Select Folder",
                    InitialDirectory = AppDomain.CurrentDomain.BaseDirectory
                };

                if (folderDialog.ShowDialog() == true)
                {
                    EnableConsole();

                    string folderName = folderDialog.FolderName;

                    //results.Add(await Task.Run(() => AI.TrainerAndValidator.Validate(CancelToken, folderName)));

                    // Calculate the average loss and accuracy from the results
                    float res1 = results.Average(result => result.Loss);
                    float res2 = results.Average(result => result.MAE);

                    // Prepare the result message
                    string message = $"Validation complete!\nAverage Loss: {res1:F4}\nAverage MAE: {res2:F4}";

                    // Determine whether the overall result is a success or error
                    bool allSuccessful = results.All(result => result.Success);
                    string title = allSuccessful ? "Success" : "Error";
                    MessageBoxImage icon = allSuccessful ? MessageBoxImage.Information : MessageBoxImage.Error;

                    // Show success/error popup with averages
                    MessageBox.Show(message, title, MessageBoxButton.OK, icon);
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
                ValidateButtonEnabled = true;
                IsBusyTraining = false;
            }
        }


        [RelayCommand]
        public void OpenImage()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.gif",
                Title = "Select an Image",
                InitialDirectory = AppDomain.CurrentDomain.BaseDirectory
            };

            if (openFileDialog.ShowDialog() == true)
            {
                string filePath = openFileDialog.FileName;
                ImageToDisplay = new BitmapImage(new Uri(filePath));

                imageURL = openFileDialog.FileName;
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

                // RUN: Training on background thread
                //AI.AIResult result = await Task.Run(() => AI.Predictor.Predict(imageURL));

                // RESULT: Show success/error popup
                //MessageBox.Show(result.Message,
                //    result.Success ? "Success" : "Error",
                //    MessageBoxButton.OK,
                //    result.Success ? MessageBoxImage.Information : MessageBoxImage.Error);
            }

            catch (Exception ex)
            {
                MessageBox.Show($"An error occurred: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                TrainingButtonEnabled = true;
                ValidateButtonEnabled = true;
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
            ValidateButtonEnabled = true;
            IsBusyTraining = false;
        }

    }

}
