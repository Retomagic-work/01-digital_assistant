import {Injectable} from "@nestjs/common";
import {UsersService} from "../users/users.service";
import * as bcrypt from "bcrypt";
import {JwtService} from "@nestjs/jwt";
import {UsersEntity} from "../users/entities/users.entity";

@Injectable()
export class AuthService {
  constructor(
    private readonly usersService: UsersService,
    private jwtService: JwtService,
  ) {
  }

  async validateUser(email: string, password: string) {
    const user = await this.usersService.getUserByLogin(email);
    if (!user) return null;

    const passwordValid = await bcrypt.compare(password, user.password);
    if (passwordValid) return user;

    return null;
  }

  async login(user: UsersEntity) {
    const payload = {username: user.login, sub: user.id};
    return {
      access_token: this.jwtService.sign(payload)
    };
  }
}
