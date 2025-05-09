using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AICircleDetector.WPF
{
    public class ConsoleBindingWriter : TextWriter
    {
        private readonly Action<string> _writeCallback;
        private readonly StringBuilder _buffer = new();

        public ConsoleBindingWriter(Action<string> writeCallback)
        {
            _writeCallback = writeCallback;
        }

        public override void Write(char value)
        {
            _buffer.Append(value);
        }

        public override void Write(string? value)
        {
            if (!string.IsNullOrWhiteSpace(value))
            {
                string normalized = value.Replace("\r\n", "\n").Replace("\r", "\n"); // Normalize all
                string[] lines = normalized.Split('\n');

                foreach (string line in lines)
                {
                    string trimmed = line.TrimEnd();
                    if (!string.IsNullOrWhiteSpace(trimmed))
                        _writeCallback(trimmed);
                }
            }
        }

        public override Encoding Encoding => Encoding.UTF8;
    }

}
