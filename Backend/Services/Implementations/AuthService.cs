using Mega.DTO;
using Mega.Models;
using Mega.Services.Interfaces;
using Microsoft.AspNetCore.Identity;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;

namespace Mega.Services.Implementations
{
    public class AuthService : IAuthService
    {
        private readonly UserManager<ApplicationUser> _userManager;
        private readonly IConfiguration _config;

        public AuthService(UserManager<ApplicationUser> userManager, IConfiguration config)
        {
            _userManager = userManager;
            _config = config;
        }

        public async Task<IdentityResult> RegisterUserAsync(RegisterDTO userFromRequest)
        {
            var user = new ApplicationUser
            {
                UserName = userFromRequest.UserName,
                Email = userFromRequest.Email,
                FullName = userFromRequest.FullName,
                PhoneNumber = userFromRequest.PhoneNumber
            };

            return await _userManager.CreateAsync(user, userFromRequest.Password);
        }

        public async Task<AuthResponseDTO> LoginUserAsync(LoginDTO userFromRequest)
        {
            var user = await _userManager.FindByNameAsync(userFromRequest.UserName)
                       ?? await _userManager.FindByEmailAsync(userFromRequest.UserName);

            if (user == null || !await _userManager.CheckPasswordAsync(user, userFromRequest.Password))
                return new AuthResponseDTO { Token = string.Empty };

            // ADD NULL CHECK for configuration
            var secretKey = _config["JWT:SecretKey"];
            if (string.IsNullOrEmpty(secretKey))
            {
                throw new InvalidOperationException("JWT:SecretKey is not configured in appsettings.json");
            }

            var claims = new List<Claim>
            {
                new(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString()),
                new(ClaimTypes.NameIdentifier, user.Id),
                new(ClaimTypes.Name, user.UserName ?? string.Empty),
                new(ClaimTypes.Email, user.Email ?? string.Empty)
            };

            var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(secretKey));
            var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

            var expirationHours = _config.GetValue<int>("JWT:ExpirationInHours", 24);
            var expirationTime = DateTime.UtcNow.AddHours(expirationHours);

            var jwtToken = new JwtSecurityToken(
                issuer: _config["JWT:Issuer"] ?? "MegaAPI",
                audience: _config["JWT:Audience"] ?? "MegaUsers",
                claims: claims,
                expires: expirationTime,
                signingCredentials: creds
            );

            var tokenString = new JwtSecurityTokenHandler().WriteToken(jwtToken);

            return new AuthResponseDTO
            {
                Token = tokenString,
                Expiration = expirationTime,
                ExpiresIn = (long)(expirationTime - DateTime.UtcNow).TotalSeconds,
                UserName = user.UserName ?? string.Empty,
                Email = user.Email ?? string.Empty,
                FullName = user.FullName ?? string.Empty,
                TokenType = "Bearer"
            };
        }
    }
}