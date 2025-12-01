using Mega.DTO;
using Microsoft.AspNetCore.Identity;

namespace Mega.Services.Interfaces
{
    public interface IAuthService
    {
        Task<IdentityResult> RegisterUserAsync(RegisterDTO userFromRequest);
        Task<AuthResponseDTO> LoginUserAsync(LoginDTO userFromRequest);
    }
}
