using Mega.DTO;
using Mega.Helper;
using Mega.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using System.Text.Json;
namespace Mega.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [Authorize]
    public class DatasetController : ControllerBase
    {
        private readonly MegaContext _context;
        private readonly UploadHelper _uploadHelper;
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly string _aiBaseUrl = "https://omarnashat2004--agentic-ai-app-fastapi-app.modal.run";
        public DatasetController(MegaContext context, UploadHelper uploadHelper, IHttpClientFactory httpClientFactory)
        {
            _context = context;
            _uploadHelper = uploadHelper;
            _httpClientFactory = httpClientFactory;
        }
        // ===================== UPLOAD FILE =====================
        // Endpoint to upload a dataset file
        [HttpPost("upload")]
        public async Task<IActionResult> Upload(
            IFormFile file,
            [FromForm] string targetColumn,
            [FromForm] bool runAutoML = false)
        {
            // Validate the target column
            if (string.IsNullOrWhiteSpace(targetColumn))
                return BadRequest(new { error = "Target column is required" });
            // Get current user ID
            var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userId))
                return Unauthorized(new { error = "User not authenticated" });
            // Retrieve the user from the database
            var user = await _context.Users.FirstOrDefaultAsync(u => u.Id == userId);
            if (user == null)
                return Unauthorized(new { error = "User not found" });
            try
            {
                // Upload the file to Cloudinary
                var uploadResult = await _uploadHelper.UploadAsync(file);
                if (!uploadResult.Success)
                    return BadRequest(new { error = uploadResult.ErrorMessage });
                // Save the record in the database
                var record = new DatasetRecord
                {
                    FileName = file.FileName,
                    FilePath = uploadResult.Url!,
                    UploadedAt = DateTime.UtcNow,
                    UserId = userId,
                    TargetColumn = targetColumn,
                    RunAutoML = runAutoML,
                };
                _context.DatasetRecords.Add(record);
                await _context.SaveChangesAsync();
                // Initialize variable to store AI response data
                object? aiData = null;
                try
                {
                    // Create HTTP client to communicate with AI server
                    var httpClient = _httpClientFactory.CreateClient();
                    // Prepare request object to send to AI server
                    var aiRequest = new
                    {
                        user_id = userId,
                        filename = file.FileName,
                        url = uploadResult.Url,
                        target_column = targetColumn,
                        run_automl = runAutoML
                    };
                    // Send POST request to AI server
                    var aiResponse = await httpClient.PostAsJsonAsync($"{_aiBaseUrl}/start", aiRequest);

                    // If AI server returns success, read and store the response
                    if (aiResponse.IsSuccessStatusCode)
                    {
                        aiData = await aiResponse.Content.ReadFromJsonAsync<object>();
                        record.AIResponseJson = JsonSerializer.Serialize(aiData);
                        await _context.SaveChangesAsync();
                    }
                    else
                    {
                        // Store error information if AI server returns failure
                        aiData = new { error = "AI server returned error", statusCode = aiResponse.StatusCode };
                    }
                }
                catch (Exception aiEx)
                {
                    // Handle exceptions while calling AI server
                    aiData = new { error = aiEx.Message, stackTrace = aiEx.StackTrace };
                }
                // Return success response to frontend
                return Ok(new
                {
                    message = "File uploaded successfully",
                    datasetId = record.Id,
                    fileName = record.FileName,
                    uploadedAt = record.UploadedAt,
                    userName = user.UserName,
                });
            }
            catch (Exception ex)
            {
                // Handle any file upload exceptions
                return StatusCode(500, new { error = $"File upload failed: {ex.Message}", stackTrace = ex.StackTrace });
            }
        }
        // ===================== GET SCHEMA =====================
        // Endpoint to get dataset schema from AI server
        [HttpGet("schema/{datasetId}")]
        public async Task<IActionResult> GetSchema(int datasetId)
        {
            try
            {
                // Get current user ID
                var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
                // Retrieve dataset record from database
                var dataset = await _context.DatasetRecords.FindAsync(datasetId);
                // Check if dataset exists
                if (dataset == null)
                    return NotFound(new { error = "Dataset not found" });
                // Ensure the dataset belongs to the authenticated user
                if (dataset.UserId != userId)
                    return Unauthorized(new { error = "Access denied: dataset belongs to another user" });
                // Check if AI response is available
                if (string.IsNullOrEmpty(dataset.AIResponseJson))
                    return BadRequest(new { error = "AI response is empty" });
                // Deserialize AI response
                var aiJson = JsonSerializer.Deserialize<JsonElement>(dataset.AIResponseJson);
                // Check if endpoints exist in AI response
                if (!aiJson.TryGetProperty("endpoints", out JsonElement endpoints))
                    return BadRequest(new { error = "AI response does not contain endpoints" });
                // Call schema endpoint from AI server
                var schemaUrl = _aiBaseUrl + endpoints.GetProperty("schema").GetString();
                var http = _httpClientFactory.CreateClient();
                var result = await http.GetStringAsync(schemaUrl);
                return Ok(JsonSerializer.Deserialize<object>(result));
            }
            catch (Exception ex)
            {
                // Handle exceptions
                return StatusCode(500, new { error = ex.Message });
            }
        }
        // ===================== GET DASHBOARD =====================
        // Endpoint to get dataset dashboard from AI server
        [HttpGet("dashboard/{datasetId}")]
        public async Task<IActionResult> GetDashboard(int datasetId)
        {
            try
            {
                // Get current user ID
                var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
                // Retrieve dataset record from database
                var dataset = await _context.DatasetRecords.FindAsync(datasetId);

                // Check if dataset exists
                if (dataset == null)
                    return NotFound(new { error = "Dataset not found" });
                // Ensure the dataset belongs to the authenticated user
                if (dataset.UserId != userId)
                    return Unauthorized(new { error = "Access denied: dataset belongs to another user" });
                // Deserialize AI response
                var aiJson = JsonSerializer.Deserialize<JsonElement>(dataset.AIResponseJson);
                // Call dashboard endpoint from AI server
                var url = _aiBaseUrl + aiJson.GetProperty("endpoints").GetProperty("dashboard").GetString();
                var http = _httpClientFactory.CreateClient();
                var result = await http.GetStringAsync(url);
                return Ok(JsonSerializer.Deserialize<object>(result));
            }
            catch (Exception ex)
            {
                // Handle exceptions
                return StatusCode(500, new { error = ex.Message });
            }
        }
        // ===================== GET MODELS =====================
        // Endpoint to get available AI models for the dataset
        [HttpGet("models/{datasetId}")]
        public async Task<IActionResult> GetModels(int datasetId)
        {
            try
            {
                // Get current user ID
                var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
                // Retrieve dataset record from database
                var dataset = await _context.DatasetRecords.FindAsync(datasetId);
                // Check if dataset exists
                if (dataset == null)
                    return NotFound(new { error = "Dataset not found" });
                // Ensure the dataset belongs to the authenticated user
                if (dataset.UserId != userId)
                    return Unauthorized(new { error = "Access denied: dataset belongs to another user" });
                // Deserialize AI response
                var aiJson = JsonSerializer.Deserialize<JsonElement>(dataset.AIResponseJson);
                // Call models endpoint from AI server
                var url = _aiBaseUrl + aiJson.GetProperty("endpoints").GetProperty("models").GetString();
                var http = _httpClientFactory.CreateClient();
                var result = await http.GetStringAsync(url);
                return Ok(JsonSerializer.Deserialize<object>(result));
            }
            catch (Exception ex)
            {
                // Handle exceptions
                return StatusCode(500, new { error = ex.Message });
            }
        }
        // ===================== RUN PREDICTION =====================
        // Endpoint to predict using AI model
        [HttpPost("predict")]
        public async Task<IActionResult> Predict([FromBody] PredictionRequestDTO request)
        {
            try
            {
                // Get current user ID
                var userId = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
                // Retrieve dataset record from database
                var dataset = await _context.DatasetRecords.FindAsync(request.DatasetId);
                // Check if dataset exists
                if (dataset == null)
                    return NotFound(new { error = "Dataset not found" });
                // Ensure the dataset belongs to the authenticated user
                if (dataset.UserId != userId)
                    return Unauthorized(new { error = "Access denied: dataset belongs to another user" });
                // Check if AI response is available
                if (string.IsNullOrEmpty(dataset.AIResponseJson))
                    return BadRequest(new { error = "AI response is empty" });
                // Deserialize AI response
                var aiJson = JsonSerializer.Deserialize<JsonElement>(dataset.AIResponseJson);
                // Check if endpoints exist in AI response
                if (!aiJson.TryGetProperty("endpoints", out JsonElement endpoints))
                    return BadRequest(new { error = "AI response missing endpoints" });
                // Predict using AI
                var predictUrl = _aiBaseUrl + endpoints.GetProperty("predict").GetString();
                var http = _httpClientFactory.CreateClient();
                // Prepare request object for AI prediction
                var aiRequest = new
                {
                    user_id = userId,
                    version = request.Version,
                    model_name = request.Model_Name,
                    features = request.Features
                };
                // Send prediction request to AI server
                var response = await http.PostAsJsonAsync(predictUrl, aiRequest);
                // Handle AI prediction errors
                if (!response.IsSuccessStatusCode)
                {
                    var error = await response.Content.ReadAsStringAsync();
                    return BadRequest(new
                    {
                        error = "AI prediction server error",
                        details = error
                    });
                }
                // Return AI prediction result
                var result = await response.Content.ReadFromJsonAsync<object>();
                return Ok(result);
            }
            catch (Exception ex)
            {
                // Handle exceptions
                return StatusCode(500, new { error = ex.Message });
            }
        }
    }
}
