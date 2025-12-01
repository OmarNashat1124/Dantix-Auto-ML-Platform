using CloudinaryDotNet;
using CloudinaryDotNet.Actions;
namespace Mega.Helper
{
    public class UploadHelper
    {
        private readonly Cloudinary _cloudinary;
        private readonly List<string> _allowedExtensions = new() { ".csv", ".json", ".xlsx" };
        private const long MaxFileSize = 20 * 1024 * 1024; // 20MB

        public UploadHelper(Cloudinary cloudinary)
        {
            _cloudinary = cloudinary ?? throw new ArgumentNullException(nameof(cloudinary));
        }

        public async Task<(bool Success, string? Url, string? ErrorCode, string? ErrorMessage)> UploadAsync(IFormFile file)
        {
            if (file == null || file.Length == 0)
                return (false, null, "NO_FILE", "No file uploaded.");

            if (file.Length > MaxFileSize)
                return (false, null, "FILE_TOO_LARGE", $"File size exceeds 20MB limit.");

            var fileExtension = Path.GetExtension(file.FileName).ToLowerInvariant();
            if (!_allowedExtensions.Contains(fileExtension))
                return (false, null, "INVALID_FILE_TYPE", $"Allowed file types: {string.Join(", ", _allowedExtensions)}");

            try
            {
                using var stream = file.OpenReadStream();
                var uploadParams = new RawUploadParams
                {
                    File = new FileDescription(file.FileName, stream),
                    Folder = "MegaProjectFiles",
                    PublicId = $"{Guid.NewGuid()}_{Path.GetFileNameWithoutExtension(file.FileName)}",
                    Overwrite = false
                };

                var uploadResult = await _cloudinary.UploadAsync(uploadParams);

                if (uploadResult.Error != null)
                {
                    return (false, null, "UPLOAD_FAILED", $"Cloudinary upload failed: {uploadResult.Error.Message}");
                }

                if (uploadResult.StatusCode == System.Net.HttpStatusCode.OK)
                {
                    return (true, uploadResult.SecureUrl.ToString(), null, null);
                }

                return (false, null, "UPLOAD_FAILED", $"Upload failed with status: {uploadResult.StatusCode}");
            }
            catch (Exception ex)
            {
                return (false, null, "EXCEPTION", $"Upload error: {ex.Message}");
            }
        }
    }
}
