using AICircleDetector.WPF.Converter;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace AICircleDetector.WPF.ViewModels
{

    public partial class MainControlViewModel : BaseViewModel
    {
        private int TrainingSetsCount = 2;

        private CancellationTokenSource CancellationTokenSource = new CancellationTokenSource();
        private CancellationToken CancelToken;

        [ObservableProperty]
        private ImageSource imageToDisplay;

        private string imageURL = string.Empty;

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

            for (int i = 0; i < TrainingSetsCount; i++)
            {
                await Task.Run(() => AI.TrainingDataBuilder.CreateTrainingData(imageCount: 10));
            }


            MessageBox.Show("Testfiles have been created successfully", "Result", MessageBoxButton.OK, MessageBoxImage.Information);

            DataButtonEnabled = true;
            IsBusyDataCreation = false;
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

            try
            {
                OpenFolderDialog folderDialog = new OpenFolderDialog
                {
                    Title = "Select Folder with training folders",
                    InitialDirectory = AppDomain.CurrentDomain.BaseDirectory
                };

                if (folderDialog.ShowDialog() == true)
                {
                    string folderName = folderDialog.FolderName;

                    // SETUP: Redirect console output to ConsoleText
                    if (false)
                        Console.SetOut(new ConsoleBindingWriter(AppendConsoleLine));

                    // Get the first-level subfolders within the selected folder
                    string[] subfolders = Directory.GetDirectories(folderName);

                    foreach (string subfolder in subfolders)
                    {
                        if (CancelToken.IsCancellationRequested)
                        {
                            MessageBox.Show("Training abourted by User", "Training Cancel", MessageBoxButton.OK, MessageBoxImage.Information);
                            return;
                        }
                        // RUN: Training on background thread
                        results.Add(await Task.Run(() => AI.Trainer.Train(CancelToken, subfolder)));
                    }

                    // Calculate the average loss and accuracy from the results
                    float averageLoss = results.Average(result => result.Loss);
                    float averageMAE = results.Average(result => result.MAE);

                    // Prepare the result message
                    string message = $"Training complete!\nAverage Loss: {averageLoss:F4}\nAverage Mean Absolute Error: {averageMAE:F4}";

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
                // SETUP: Redirect console output to ConsoleText
                Console.SetOut(new ConsoleBindingWriter(AppendConsoleLine));

                // RUN: Training on background thread
                AI.AIResult result = await Task.Run(() => AI.Predictor.Predict(imageURL));

                // RESULT: Show success/error popup
                MessageBox.Show(result.Message,
                    result.Success ? "Success" : "Error",
                    MessageBoxButton.OK,
                    result.Success ? MessageBoxImage.Information : MessageBoxImage.Error);
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
        }

    }

}
