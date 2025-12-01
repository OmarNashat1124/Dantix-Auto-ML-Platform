using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace Mega.Models
{
    public class MegaContext : IdentityDbContext<ApplicationUser>
    {
        public MegaContext(DbContextOptions<MegaContext> options) : base(options)
        {
        }
        public DbSet<DatasetRecord> DatasetRecords { get; set; }
    }
}