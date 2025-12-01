using Mega.Models;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

public class DatasetRecord
{
    public int Id { get; set; }

    [Required, MaxLength(200)]
    public string FileName { get; set; }

    [Required, MaxLength(500)]
    public string FilePath { get; set; }

    [Required]
    public DateTime UploadedAt { get; set; } = DateTime.UtcNow;

    [Required]
    public string UserId { get; set; }

    [ForeignKey("UserId")]
    public virtual ApplicationUser User { get; set; }

    [Required]
    public string TargetColumn { get; set; }

    [Required]
    public bool RunAutoML { get; set; } = false;

    [Required]
    public string AIResponseJson { get; set; } = string.Empty;
}
