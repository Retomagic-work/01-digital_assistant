import {Body, Controller, Post, Request, UseGuards} from "@nestjs/common";
import {AuthService} from "./auth.service";
import {LocalAuthGuard} from "./guards/local-auth.guard";
import {LoginUsersDto} from "../users/dto/login.users.dto";
import {ApiOperation, ApiTags} from "@nestjs/swagger";

@ApiTags("Авторизация")
@Controller("auth")
export class AuthController {
  constructor(private readonly authService: AuthService) {
  }

  @ApiOperation({summary: "Авторизация"})
  @UseGuards(LocalAuthGuard)
  @Post("/login")
  async login(@Request() req: Request, @Body() body: LoginUsersDto) {
    return this.authService.login(req["user"]);
  }
}
