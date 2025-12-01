using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Mega.Migrations
{
    /// <inheritdoc />
    public partial class RunAutoML : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "RunAutoML",
                table: "DatasetRecords",
                type: "bit",
                nullable: false,
                defaultValue: false);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "RunAutoML",
                table: "DatasetRecords");
        }
    }
}
