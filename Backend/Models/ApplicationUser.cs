using Microsoft.AspNetCore.Identity;
using System.ComponentModel.DataAnnotations;

namespace Mega.Models
{
    public class ApplicationUser : IdentityUser
    {
        [Required]
        [StringLength(100)]
        public string FullName { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public virtual ICollection<DatasetRecord> DatasetRecords { get; set; } = new List<DatasetRecord>();
    }
}
