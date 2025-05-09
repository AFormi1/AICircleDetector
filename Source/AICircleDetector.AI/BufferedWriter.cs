
namespace AICircleDetector.AI
{
    public class BufferedWriter : IDisposable
    {
        private FileStream _fileStream;

        public BufferedWriter(string filePath)
        {
            _fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        }

        public void Write(byte[] data)
        {
            _fileStream.Write(data, 0, data.Length);
        }

        public void Dispose()
        {
            _fileStream?.Dispose();
        }
    }
}