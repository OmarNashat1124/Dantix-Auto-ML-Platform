using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Mega.Migrations
{
    /// <inheritdoc />
    public partial class v3 : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "AiDashboardUrl",
                table: "DatasetRecords",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "AiModelsUrl",
                table: "DatasetRecords",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "AiSchemaUrl",
                table: "DatasetRecords",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "AiUserId",
                table: "DatasetRecords",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<int>(
                name: "AiVersion",
                table: "DatasetRecords",
                type: "int",
                nullable: false,
                defaultValue: 0);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "AiDashboardUrl",
                table: "DatasetRecords");

            migrationBuilder.DropColumn(
                name: "AiModelsUrl",
                table: "DatasetRecords");

            migrationBuilder.DropColumn(
                name: "AiSchemaUrl",
                table: "DatasetRecords");

            migrationBuilder.DropColumn(
                name: "AiUserId",
                table: "DatasetRecords");

            migrationBuilder.DropColumn(
                name: "AiVersion",
                table: "DatasetRecords");
        }
    }
}
