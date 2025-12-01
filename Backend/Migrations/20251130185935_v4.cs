using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Mega.Migrations
{
    /// <inheritdoc />
    public partial class v4 : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_DatasetRecords_AspNetUsers_UserId",
                table: "DatasetRecords");

            migrationBuilder.DropIndex(
                name: "IX_DatasetRecords_UserId",
                table: "DatasetRecords");

            migrationBuilder.DropColumn(
                name: "AIResponseJson",
                table: "DatasetRecords");

            migrationBuilder.AlterColumn<string>(
                name: "UserId",
                table: "DatasetRecords",
                type: "nvarchar(max)",
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(450)");

            migrationBuilder.AlterColumn<string>(
                name: "FilePath",
                table: "DatasetRecords",
                type: "nvarchar(max)",
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(500)",
                oldMaxLength: 500);

            migrationBuilder.AlterColumn<string>(
                name: "FileName",
                table: "DatasetRecords",
                type: "nvarchar(max)",
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(200)",
                oldMaxLength: 200);

            migrationBuilder.AddColumn<string>(
                name: "ApplicationUserId",
                table: "DatasetRecords",
                type: "nvarchar(450)",
                nullable: true);

            migrationBuilder.CreateIndex(
                name: "IX_DatasetRecords_ApplicationUserId",
                table: "DatasetRecords",
                column: "ApplicationUserId");

            migrationBuilder.AddForeignKey(
                name: "FK_DatasetRecords_AspNetUsers_ApplicationUserId",
                table: "DatasetRecords",
                column: "ApplicationUserId",
                principalTable: "AspNetUsers",
                principalColumn: "Id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_DatasetRecords_AspNetUsers_ApplicationUserId",
                table: "DatasetRecords");

            migrationBuilder.DropIndex(
                name: "IX_DatasetRecords_ApplicationUserId",
                table: "DatasetRecords");

            migrationBuilder.DropColumn(
                name: "ApplicationUserId",
                table: "DatasetRecords");

            migrationBuilder.AlterColumn<string>(
                name: "UserId",
                table: "DatasetRecords",
                type: "nvarchar(450)",
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(max)");

            migrationBuilder.AlterColumn<string>(
                name: "FilePath",
                table: "DatasetRecords",
                type: "nvarchar(500)",
                maxLength: 500,
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(max)");

            migrationBuilder.AlterColumn<string>(
                name: "FileName",
                table: "DatasetRecords",
                type: "nvarchar(200)",
                maxLength: 200,
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(max)");

            migrationBuilder.AddColumn<string>(
                name: "AIResponseJson",
                table: "DatasetRecords",
                type: "nvarchar(max)",
                nullable: true);

            migrationBuilder.CreateIndex(
                name: "IX_DatasetRecords_UserId",
                table: "DatasetRecords",
                column: "UserId");

            migrationBuilder.AddForeignKey(
                name: "FK_DatasetRecords_AspNetUsers_UserId",
                table: "DatasetRecords",
                column: "UserId",
                principalTable: "AspNetUsers",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }
    }
}
