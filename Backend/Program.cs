using Mega.Models;
using Mega.Services.Implementations;
using Mega.Services.Interfaces;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using Microsoft.OpenApi.Models;
using System.Text;
using Mega.Helper;
using CloudinaryDotNet;

var builder = WebApplication.CreateBuilder(args);

// ------------------- Add services ------------------- //
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddHttpClient(); // for AI integration

// ---------- Database Configuration ---------- //
builder.Services.AddDbContext<MegaContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection") ??
        throw new InvalidOperationException("Connection string 'DefaultConnection' not found.")));

// ---------- Identity Configuration ---------- //
builder.Services.AddIdentity<ApplicationUser, IdentityRole>(options =>
{
    options.Password.RequireDigit = true;
    options.Password.RequireLowercase = true;
    options.Password.RequireUppercase = true;
    options.Password.RequireNonAlphanumeric = true;
    options.Password.RequiredLength = 8;
    options.User.RequireUniqueEmail = true;
})
.AddEntityFrameworkStores<MegaContext>()
.AddDefaultTokenProviders();

// ---------- Cloudinary Configuration ---------- //
builder.Services.Configure<CloudinarySettings>(builder.Configuration.GetSection("Cloudinary"));
builder.Services.AddSingleton(provider =>
{
    var settings = builder.Configuration.GetSection("Cloudinary");
    var cloudName = settings["CloudName"];
    var apiKey = settings["ApiKey"];
    var apiSecret = settings["ApiSecret"];

    if (string.IsNullOrEmpty(cloudName) || string.IsNullOrEmpty(apiKey) || string.IsNullOrEmpty(apiSecret))
        throw new InvalidOperationException("Cloudinary configuration is incomplete");

    var account = new Account(cloudName, apiKey, apiSecret);
    return new Cloudinary(account);
});
// ---------- Services ---------- //
builder.Services.AddScoped<IAuthService, AuthService>();
builder.Services.AddScoped<UploadHelper>();

// ---------- JWT Authentication ---------- //
var jwtSecretKey = builder.Configuration["JWT:SecretKey"] ??
    throw new InvalidOperationException("JWT:SecretKey is not configured");
var jwtIssuer = builder.Configuration["JWT:Issuer"] ?? "https://localhost:5263";
var jwtAudience = builder.Configuration["JWT:Audience"] ?? "https://localhost:4200";
builder.Services.AddAuthentication(options =>
{
    options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultScheme = JwtBearerDefaults.AuthenticationScheme;
})
.AddJwtBearer(options =>
{
    options.SaveToken = true;
    options.RequireHttpsMetadata = true;
    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuer = true,
        ValidIssuer = jwtIssuer,
        ValidateAudience = true,
        ValidAudience = jwtAudience,
        ValidateIssuerSigningKey = true,
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecretKey)),
        ValidateLifetime = true,
        ClockSkew = TimeSpan.Zero
    };
});
// ---------- CORS (Allow All) ---------- //
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

// ---------- Swagger ---------- //
builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new OpenApiInfo
    {
        Title = "Mega API",
        Version = "v1",
        Description = "Mega Project API with JWT Authentication and Cloudinary"
    });

    options.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
    {
        Name = "Authorization",
        Type = SecuritySchemeType.ApiKey,
        Scheme = "Bearer",
        BearerFormat = "JWT",
        In = ParameterLocation.Header,
        Description = "Enter 'Bearer' [space] and then your token"
    });
    options.AddSecurityRequirement(new OpenApiSecurityRequirement
    {
        {
            new OpenApiSecurityScheme
            {
                Reference = new OpenApiReference
                {
                    Type = ReferenceType.SecurityScheme,
                    Id = "Bearer"
                }
            },
            Array.Empty<string>()
        }
    });
});
var app = builder.Build();
// ------------------- Middleware ------------------- //
app.UseStaticFiles();
app.UseCors("AllowAll");           // Apply global CORS policy
app.UseAuthentication();
app.UseAuthorization();
// Middleware: handle 404 routes
app.Use(async (context, next) =>
{
    await next();

    if (context.Response.StatusCode == 404 && !context.Response.HasStarted)
    {
        context.Response.ContentType = "application/json";
        await context.Response.WriteAsJsonAsync(new
        {
            error = "Route not found",
            statusCode = 404
        });
    }
});
// Middleware: handle global exceptions
app.Use(async (context, next) =>
{
    try
    {
        await next();
    }
    catch (Exception ex)
    {
        context.Response.StatusCode = 500;
        context.Response.ContentType = "application/json";
        await context.Response.WriteAsJsonAsync(new
        {
            error = ex.Message,
            stackTrace = ex.StackTrace
        });
    }
});
  
app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "Mega API V1");
    c.RoutePrefix = string.Empty; // Swagger at root
});
app.MapControllers();
app.Run();
